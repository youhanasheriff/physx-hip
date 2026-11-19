// Copyright (c) 2026 Youhana Sheriff
// SPDX-License-Identifier: Apache-2.0
//
// Runs the same rigid-body scenes on PhysX's CPU pipeline and on its GPU pipeline and
// compares the results.
//
// This file is part of the HIP port and is not part of upstream PhysX.
//
// There is no NVIDIA GPU to compare against, so the CPU pipeline is the reference. The two
// pipelines use different solvers orderings and are not bit-identical even on NVIDIA
// hardware, so each scene is compared with a tolerance that fits how sensitive it is:
// tight for ballistic motion and bodies at rest, loose and statistical for chaotic impacts.
//
// usage: cpu_gpu_compare [--tgs] [--verbose]
// exit code 0 when every scene passes.

#include "PxPhysicsAPI.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace physx;

namespace
{

PxDefaultAllocator gAllocator;

// Counts what PhysX reports, so a GPU run that silently fell back to the CPU or hit a
// kernel failure is caught even when the numbers look plausible.
class CountingErrorCallback : public PxErrorCallback
{
public:
	int errors = 0;
	int warnings = 0;
	bool verbose = false;
	void reportError(PxErrorCode::Enum code, const char* message, const char* file, int line) override
	{
		const bool isError = (code & (PxErrorCode::eINTERNAL_ERROR | PxErrorCode::eABORT | PxErrorCode::eOUT_OF_MEMORY | PxErrorCode::eINVALID_OPERATION | PxErrorCode::eINVALID_PARAMETER)) != 0;
		if (isError) ++errors; else ++warnings;
		if (verbose || isError || errors + warnings <= 5)
			std::printf("    [physx %s] %s (%s:%d)\n", isError ? "error" : "warning", message, file, line);
	}
};

CountingErrorCallback gErrorCallback;
PxFoundation* gFoundation = nullptr;
PxPhysics* gPhysics = nullptr;
PxCudaContextManager* gCudaContextManager = nullptr;
PxMaterial* gMaterial = nullptr;

struct BodyState
{
	PxTransform pose;
	PxVec3 linearVelocity;
	PxVec3 angularVelocity;
};

struct Scene
{
	const char* name;
	void (*build)(PxScene&);
	PxU32 steps;
	// Per-body limits on the CPU/GPU difference. A negative limit means "not compared".
	PxReal maxPositionError;   // metres
	PxReal maxRotationError;   // radians
	// Aggregate limits, for scenes where individual bodies legitimately diverge.
	PxReal maxCentreOfMassError; // metres
	// Solver position iterations per body. The CPU and GPU solvers order constraints
	// differently, so at the default count they stop at slightly different answers; at a
	// high count both should converge on the same one, which is the sharper test.
	PxU32 positionIterations;
};

PxU32 gPositionIterations = 4;

PxRigidDynamic* addBody(PxScene& scene, const PxTransform& pose, const PxGeometry& geometry, PxReal density, const PxVec3& linVel = PxVec3(0), const PxVec3& angVel = PxVec3(0))
{
	PxRigidDynamic* body = PxCreateDynamic(*gPhysics, pose, geometry, *gMaterial, density);
	body->setLinearVelocity(linVel);
	body->setAngularVelocity(angVel);
	body->setSolverIterationCounts(gPositionIterations, 1);
	scene.addActor(*body);
	return body;
}

void addGround(PxScene& scene)
{
	scene.addActor(*PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial));
}

// Bodies in free flight: no contacts, so this isolates integration.
void buildBallistic(PxScene& scene)
{
	for (PxU32 i = 0; i < 64; ++i)
	{
		const PxReal f = PxReal(i);
		// An 8 x 8 grid, 15 m apart: far enough that no two bodies can meet, so no contact is
		// ever generated, and centred on the origin to keep coordinates small. The GPU TGS
		// solver adds each sub-step's motion to the position separately, so its rounding
		// error grows with the size of the coordinate; far from the origin that alone
		// reaches centimetres after a few hundred steps.
		const PxVec3 position((PxReal(i % 8) - 3.5f) * 15.0f, 50.0f + PxSin(f) * 5.0f, (PxReal(i / 8) - 3.5f) * 15.0f);
		const PxVec3 linVel(PxSin(f * 1.3f) * 2.0f, 5.0f + PxCos(f) * 3.0f, PxSin(f * 0.4f));
		const PxVec3 angVel(PxCos(f * 0.9f), PxSin(f * 0.3f) * 2.0f, PxCos(f * 1.7f));
		if (i % 3 == 0)
			addBody(scene, PxTransform(position), PxBoxGeometry(0.5f, 0.3f, 0.2f), 10.0f, linVel, angVel);
		else if (i % 3 == 1)
			addBody(scene, PxTransform(position), PxSphereGeometry(0.4f), 10.0f, linVel, angVel);
		else
			addBody(scene, PxTransform(position), PxCapsuleGeometry(0.2f, 0.5f), 10.0f, linVel, angVel);
	}
}

// One sphere falling from rest with nothing to hit: the smallest possible integration check.
void buildFreefall(PxScene& scene)
{
	addBody(scene, PxTransform(PxVec3(0, 100, 0)), PxSphereGeometry(0.5f), 10.0f);
}

// Separated bodies dropped a short way onto the ground: broad phase, narrow phase for
// each shape type against a plane, contact solve, and coming to rest.
void buildDropAndSettle(PxScene& scene)
{
	addGround(scene);
	for (PxU32 i = 0; i < 10; ++i)
	{
		for (PxU32 j = 0; j < 10; ++j)
		{
			const PxVec3 position(PxReal(i) * 4.0f, 1.5f, PxReal(j) * 4.0f);
			const PxU32 kind = (i + j) % 3;
			if (kind == 0)
				addBody(scene, PxTransform(position), PxBoxGeometry(0.5f, 0.5f, 0.5f), 10.0f);
			else if (kind == 1)
				addBody(scene, PxTransform(position), PxSphereGeometry(0.5f), 10.0f);
			else
				addBody(scene, PxTransform(position, PxQuat(PxHalfPi, PxVec3(0, 0, 1))), PxCapsuleGeometry(0.3f, 0.4f), 10.0f);
		}
	}
}

// A pyramid of boxes that should simply stay standing: box-box contact manifolds and a
// solver that has to hold a stack up.
void buildPyramid(PxScene& scene)
{
	addGround(scene);
	const PxU32 size = 12;
	const PxReal h = 0.5f;
	for (PxU32 i = 0; i < size; ++i)
		for (PxU32 j = 0; j < size - i; ++j)
			addBody(scene, PxTransform(PxVec3(PxReal(j * 2) - PxReal(size - i), PxReal(i * 2 + 1), 0) * h), PxBoxGeometry(h, h, h), 10.0f);
}

// A heavy ball thrown into a wall of pyramids: the HelloGRB scene at reduced size. Chaotic,
// so only aggregates are compared.
void buildImpact(PxScene& scene)
{
	addGround(scene);
	const PxU32 size = 10;
	const PxReal h = 0.5f;
	for (PxU32 s = 0; s < 4; ++s)
		for (PxU32 i = 0; i < size; ++i)
			for (PxU32 j = 0; j < size - i; ++j)
				addBody(scene, PxTransform(PxVec3(PxReal(j * 2) - PxReal(size - i), PxReal(i * 2 + 1), -PxReal(s) * 4.0f) * h), PxBoxGeometry(h, h, h), 10.0f);
	PxRigidDynamic* ball = addBody(scene, PxTransform(PxVec3(0, 6, 30)), PxSphereGeometry(1.5f), 100.0f, PxVec3(0, -3, -40));
	PX_UNUSED(ball);
}

// A hanging chain on spherical joints, swinging: joint constraints in the GPU solver.
void buildChain(PxScene& scene)
{
	const PxU32 links = 12;
	const PxReal halfLength = 0.5f;
	PxRigidActor* previous = nullptr;
	PxRigidStatic* anchor = gPhysics->createRigidStatic(PxTransform(PxVec3(0, 20, 0)));
	scene.addActor(*anchor);
	previous = anchor;
	PxVec3 previousAttach(0, 0, 0);
	for (PxU32 i = 0; i < links; ++i)
	{
		// laid out horizontally so that it swings down
		const PxVec3 centre(halfLength + PxReal(i) * 2.0f * halfLength, 20.0f, 0.0f);
		PxRigidDynamic* link = addBody(scene, PxTransform(centre), PxBoxGeometry(halfLength, 0.1f, 0.1f), 5.0f);
		PxSphericalJointCreate(*gPhysics, previous, PxTransform(previousAttach), link, PxTransform(PxVec3(-halfLength, 0, 0)));
		previous = link;
		previousAttach = PxVec3(halfLength, 0, 0);
	}
}

std::vector<BodyState> runScene(const Scene& sceneDef, bool gpu, bool tgs, double& secondsPerStep)
{
	PxSceneDesc desc(gPhysics->getTolerancesScale());
	desc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	PxDefaultCpuDispatcher* dispatcher = PxDefaultCpuDispatcherCreate(4);
	desc.cpuDispatcher = dispatcher;
	desc.filterShader = PxDefaultSimulationFilterShader;
	desc.flags |= PxSceneFlag::eENABLE_PCM;
	desc.solverType = tgs ? PxSolverType::eTGS : PxSolverType::ePGS;
	if (gpu)
	{
		desc.cudaContextManager = gCudaContextManager;
		desc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
		desc.broadPhaseType = PxBroadPhaseType::eGPU;
	}

	PxScene* scene = gPhysics->createScene(desc);
	gPositionIterations = sceneDef.positionIterations;
	sceneDef.build(*scene);

	const PxU64 start = PxTime::getCurrentCounterValue();
	for (PxU32 i = 0; i < sceneDef.steps; ++i)
	{
		scene->simulate(1.0f / 60.0f);
		scene->fetchResults(true);
	}
	const PxU64 end = PxTime::getCurrentCounterValue();
	secondsPerStep = double(PxTime::getCounterFrequency().toTensOfNanos(end - start)) * 1e-8 / double(sceneDef.steps);

	std::vector<BodyState> states;
	const PxU32 count = scene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	std::vector<PxActor*> actors(count);
	scene->getActors(PxActorTypeFlag::eRIGID_DYNAMIC, actors.data(), count);
	for (PxActor* actor : actors)
	{
		PxRigidDynamic* body = actor->is<PxRigidDynamic>();
		BodyState state;
		state.pose = body->getGlobalPose();
		state.linearVelocity = body->getLinearVelocity();
		state.angularVelocity = body->getAngularVelocity();
		states.push_back(state);
	}

	scene->release();
	dispatcher->release();
	return states;
}

bool isFinite(const BodyState& s)
{
	return s.pose.p.isFinite() && s.pose.q.isFinite() && s.linearVelocity.isFinite() && s.angularVelocity.isFinite();
}

bool compareScene(const Scene& sceneDef, bool tgs)
{
	gErrorCallback.errors = gErrorCallback.warnings = 0;
	double cpuSeconds = 0, gpuSeconds = 0;
	const std::vector<BodyState> cpu = runScene(sceneDef, false, tgs, cpuSeconds);
	const int cpuReports = gErrorCallback.errors + gErrorCallback.warnings;
	const std::vector<BodyState> gpu = runScene(sceneDef, true, tgs, gpuSeconds);
	const int gpuReports = gErrorCallback.errors + gErrorCallback.warnings - cpuReports;

	bool ok = true;
	std::printf("%-18s %4u bodies %4u steps  cpu %7.2f ms/step  gpu %7.2f ms/step\n", sceneDef.name, PxU32(cpu.size()), sceneDef.steps, cpuSeconds * 1e3, gpuSeconds * 1e3);

	if (cpu.size() != gpu.size() || cpu.empty())
	{
		std::printf("    FAIL body count differs: cpu %zu, gpu %zu\n", cpu.size(), gpu.size());
		return false;
	}
	// Any report during the GPU run fails the scene: the GPU pipeline falling back to the
	// CPU is reported as a warning, and would otherwise make the comparison trivially pass.
	if (gpuReports != 0)
	{
		std::printf("    FAIL PhysX reported %d problem(s) during the GPU run\n", gpuReports);
		ok = false;
	}

	PxReal maxPos = 0, maxRot = 0;
	PxVec3 cpuCentre(0), gpuCentre(0);
	PxReal cpuMinY = PX_MAX_F32, gpuMinY = PX_MAX_F32;
	PxU32 nonFinite = 0;
	for (size_t i = 0; i < cpu.size(); ++i)
	{
		if (!isFinite(gpu[i]) || !isFinite(cpu[i]))
		{
			++nonFinite;
			continue;
		}
		maxPos = PxMax(maxPos, (cpu[i].pose.p - gpu[i].pose.p).magnitude());
		const PxQuat delta = cpu[i].pose.q.getConjugate() * gpu[i].pose.q;
		PxReal angle = 2.0f * PxAcos(PxMin(1.0f, PxAbs(delta.w)));
		maxRot = PxMax(maxRot, angle);
		cpuCentre += cpu[i].pose.p;
		gpuCentre += gpu[i].pose.p;
		cpuMinY = PxMin(cpuMinY, cpu[i].pose.p.y);
		gpuMinY = PxMin(gpuMinY, gpu[i].pose.p.y);
	}
	cpuCentre *= 1.0f / PxReal(cpu.size());
	gpuCentre *= 1.0f / PxReal(cpu.size());
	const PxReal centreError = (cpuCentre - gpuCentre).magnitude();

	std::printf("    max position diff %.6f m, max rotation diff %.6f rad, centre-of-mass diff %.6f m, lowest body cpu %.3f gpu %.3f\n",
		double(maxPos), double(maxRot), double(centreError), double(cpuMinY), double(gpuMinY));

	// Nothing may end up below the ground plane (bodies rest at half their height or more).
	if (gpuMinY < cpuMinY - 0.1f)
	{
		std::printf("    FAIL a body sank on the GPU: lowest %.3f, CPU lowest %.3f\n", double(gpuMinY), double(cpuMinY));
		ok = false;
	}
	if (nonFinite)
	{
		std::printf("    FAIL %u bodies have non-finite state\n", nonFinite);
		ok = false;
	}
	if (sceneDef.maxPositionError >= 0 && maxPos > sceneDef.maxPositionError)
	{
		std::printf("    FAIL position diff above %.4f m\n", double(sceneDef.maxPositionError));
		ok = false;
	}
	if (sceneDef.maxRotationError >= 0 && maxRot > sceneDef.maxRotationError)
	{
		std::printf("    FAIL rotation diff above %.4f rad\n", double(sceneDef.maxRotationError));
		ok = false;
	}
	if (sceneDef.maxCentreOfMassError >= 0 && centreError > sceneDef.maxCentreOfMassError)
	{
		std::printf("    FAIL centre-of-mass diff above %.4f m\n", double(sceneDef.maxCentreOfMassError));
		ok = false;
	}
	std::printf("    %s\n", ok ? "PASS" : "FAIL");
	return ok;
}

} // namespace

int main(int argc, char** argv)
{
	bool tgs = false;
	for (int i = 1; i < argc; ++i)
	{
		if (!std::strcmp(argv[i], "--tgs")) tgs = true;
		if (!std::strcmp(argv[i], "--verbose")) gErrorCallback.verbose = true;
	}

	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), false, nullptr);
	PxInitExtensions(*gPhysics, nullptr);
	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.1f);

	PxCudaContextManagerDesc cudaDesc;
	gCudaContextManager = PxCreateCudaContextManager(*gFoundation, cudaDesc, nullptr);
	if (!gCudaContextManager || !gCudaContextManager->contextIsValid())
	{
		std::printf("No usable GPU context: the GPU pipeline cannot be tested.\n");
		return 2;
	}
	std::printf("GPU: %s, %d compute units, %.1f GiB\nsolver: %s\n\n", gCudaContextManager->getDeviceName(), gCudaContextManager->getMultiprocessorCount(),
		double(gCudaContextManager->getDeviceTotalMemBytes()) / double(1 << 30), tgs ? "TGS" : "PGS");

	const Scene scenes[] = {
		// name              build               steps  pos     rot    centre  iterations
		{"freefall",         buildFreefall,      120,   1e-3f,  1e-3f, 1e-3f,  4},
		{"ballistic",        buildBallistic,     120,   2e-3f,  2e-3f, 1e-3f,  4},
		{"drop-and-settle",  buildDropAndSettle, 240,   2e-2f,  -1.0f, 5e-3f,  4},
		// At the default iteration count the two solvers sag a stack by different amounts.
		{"pyramid",          buildPyramid,       240,   0.15f,  5e-2f, 5e-2f,  4},
		{"pyramid-converged", buildPyramid,      240,   2e-2f,  1e-2f, 5e-3f,  64},
		{"chain",            buildChain,         240,   -1.0f,  -1.0f, 0.25f,  4},
		{"chain-converged",  buildChain,         240,   5e-2f,  5e-2f, 2e-2f,  64},
		// Chaotic: a small difference in the first contact changes where everything lands.
		// Only checked for finite state, nothing below the ground, and no PhysX errors.
		{"impact",           buildImpact,        240,   -1.0f,  -1.0f, -1.0f,  4},
	};

	int failed = 0;
	for (const Scene& scene : scenes)
		if (!compareScene(scene, tgs))
			++failed;

	std::printf("\n%d of %d scenes failed\n", failed, int(sizeof(scenes) / sizeof(scenes[0])));

	PxCloseExtensions();
	gPhysics->release();
	gCudaContextManager->release();
	gFoundation->release();
	return failed == 0 ? 0 : 1;
}
