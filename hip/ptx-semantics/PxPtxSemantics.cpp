// Copyright (c) 2026 Youhana Sheriff
// SPDX-License-Identifier: Apache-2.0
//
// LLVM pass plugin that gives PhysX kernels the integer shift behaviour of NVIDIA's PTX
// when they are compiled for AMD GPUs.
//
// This file is part of the HIP port and is not part of upstream PhysX.
//
// In C++, shifting a 32-bit value by 32 or more is undefined. PhysX's kernels do it anyway
// and rely on what NVIDIA GPUs do: PTX clamps the shift amount to the register width, so an
// oversized left shift or unsigned right shift yields 0, and an oversized signed right
// shift yields the sign fill. AMD GPUs (like x86) use only the low bits of the amount, so
// the same code computes something else. The radix sort is one example: it builds
// per-digit counters with 1u << ((radix - i) << 3) and needs the out-of-range terms to
// vanish.
//
// Rather than hunting for every such expression in 90,000 lines of kernel source, this
// pass rewrites every shift whose amount is not a small-enough constant, before any
// optimisation runs:
//
//   shl  a, n   ->  n < width ? shl a, n  : 0
//   lshr a, n   ->  n < width ? lshr a, n : 0
//   ashr a, n   ->  ashr a, min(n, width - 1)
//
// Load it into the device compile with -fpass-plugin=<this library>.

#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

namespace
{

struct PxPtxSemanticsPass : PassInfoMixin<PxPtxSemanticsPass>
{
	PreservedAnalyses run(Function& F, FunctionAnalysisManager&)
	{
		SmallVector<BinaryOperator*, 32> shifts;
		for (Instruction& I : instructions(F))
		{
			auto* BO = dyn_cast<BinaryOperator>(&I);
			if (!BO || !BO->isShift())
				continue;
			const unsigned width = BO->getType()->getScalarSizeInBits();
			// A constant amount below the width (scalar, or every element of a vector) is already well defined.
			if (auto* C = dyn_cast<Constant>(BO->getOperand(1)))
			{
				if (auto* CI = dyn_cast<ConstantInt>(C))
				{
					if (CI->getValue().ult(width))
						continue;
				}
				else if (auto* Splat = dyn_cast_or_null<ConstantInt>(C->getSplatValue()))
				{
					if (Splat->getValue().ult(width))
						continue;
				}
			}
			shifts.push_back(BO);
		}

		for (BinaryOperator* BO : shifts)
		{
			Type* Ty = BO->getType();
			const unsigned width = Ty->getScalarSizeInBits();
			Value* amount = BO->getOperand(1);
			Constant* widthC = ConstantInt::get(Ty, width);

			if (BO->getOpcode() == Instruction::AShr)
			{
				IRBuilder<> B(BO);
				Value* inRange = B.CreateICmpULT(amount, widthC);
				Value* clamped = B.CreateSelect(inRange, amount, ConstantInt::get(Ty, width - 1));
				BO->setOperand(1, clamped);
			}
			else
			{
				IRBuilder<> B(BO->getNextNode());
				Value* inRange = B.CreateICmpULT(amount, widthC);
				Value* result = B.CreateSelect(inRange, BO, Constant::getNullValue(Ty));
				// Every user of the shift gets the guarded value, except the guard itself.
				BO->replaceUsesWithIf(result, [result](Use& U) { return U.getUser() != result; });
			}
		}

		return shifts.empty() ? PreservedAnalyses::all() : PreservedAnalyses::none();
	}

	// Must also run on functions marked optnone, and at -O0.
	static bool isRequired() { return true; }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo()
{
	return {LLVM_PLUGIN_API_VERSION, "PxPtxSemantics", "1.0", [](PassBuilder& PB) {
		PB.registerPipelineStartEPCallback([](ModulePassManager& MPM, OptimizationLevel) {
			MPM.addPass(createModuleToFunctionPassAdaptor(PxPtxSemanticsPass()));
		});
	}};
}
