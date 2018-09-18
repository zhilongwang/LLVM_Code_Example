//https://github.com/songlh/PDebloating/blob/master/lib/Debloater/Debloater.cpp#L795
#include <map>
#include <vector>

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/SimplifyInstructions.h"

#include "Commons/Search/Search.h"
#include "Debloater/Debloater.h"


using namespace llvm;
using namespace std;

static RegisterPass<Debloater> X(
        "remove-struct-RW",
        "remove struct RW",
        false, false);


static cl::opt<unsigned> uLoopSrcLine("noLoopLine", 
					cl::desc("Source Code Line Number for the Loop"), cl::Optional, 
					cl::value_desc("uLoopCodeLine"));

static cl::opt<std::string> strFuncName("strFunc", 
					cl::desc("Function Name"), cl::Optional, 
					cl::value_desc("strFuncName"));

/*
Value * DecomposeGEPExpression(const Value *V, uint64_t &BaseOffs, DataLayout * TD)
{
	BaseOffs = 0;
	unsigned MaxLoopup = 6;

	do
	{
		const Operator * Op = dyn_cast<Operator>(V);

		if(!Op)
		{
			if(const GlobalAlias *GA = dyn_cast<GlobalAlias>(V))
			{	
				if(!GA->isInterposable())
				{
					V = GA->getAliasee();
					continue;
				}
			}

			return V;
		}

		if(Op->getOpcode() == Instruction::BitCast || Op->getOpcode() == Instruction::AddrSpaceCast)
		{
			V = Op->getOperand(0);
			continue;
		}

		const GEPOperator * GEPOp = dyn_cast<GEPOperator>(Op);

		if(!GEPOp)
		{
			return V;
		}

		if(!GEPOp->getSourceElementType()->isSized())
		{
			return V;
		}


	} while(--MaxLoopup);
}

*/

static const unsigned MaxLookupSearchDepth = 6;

struct VariableGEPIndex {

    // An opaque Value - we can't decompose this further.
    const Value *V;

    // We need to track what extensions we've done as we consider the same Value
    // with different extensions as different variables in a GEP's linear
    // expression;
    // e.g.: if V == -1, then sext(x) != zext(x).
    unsigned ZExtBits;
    unsigned SExtBits;

    int64_t Scale;

    bool operator==(const VariableGEPIndex &Other) const {
      return V == Other.V && ZExtBits == Other.ZExtBits &&
             SExtBits == Other.SExtBits && Scale == Other.Scale;
    }

    bool operator!=(const VariableGEPIndex &Other) const {
      return !operator==(Other);
    }
};


struct DecomposedGEP {
    // Base pointer of the GEP
    const Value *Base;
    // Total constant offset w.r.t the base from indexing into structs
    int64_t StructOffset;
    // Total constant offset w.r.t the base from indexing through
    // pointers/arrays/vectors
    int64_t OtherOffset;
    // Scaled variable (non-constant) indices.
    SmallVector<VariableGEPIndex, 4> VarIndices;
};

const Value * GetLinearExpression(
    const Value *V, APInt &Scale, APInt &Offset, unsigned &ZExtBits,
    unsigned &SExtBits, const DataLayout &DL, unsigned Depth,
    AssumptionCache *AC, DominatorTree *DT, bool &NSW, bool &NUW) {
  assert(V->getType()->isIntegerTy() && "Not an integer value");

  // Limit our recursion depth.
  if (Depth == 6) {
    Scale = 1;
    Offset = 0;
    return V;
  }

  if (const ConstantInt *Const = dyn_cast<ConstantInt>(V)) {
    // If it's a constant, just convert it to an offset and remove the variable.
    // If we've been called recursively, the Offset bit width will be greater
    // than the constant's (the Offset's always as wide as the outermost call),
    // so we'll zext here and process any extension in the isa<SExtInst> &
    // isa<ZExtInst> cases below.
    Offset += Const->getValue().zextOrSelf(Offset.getBitWidth());
    assert(Scale == 0 && "Constant values don't have a scale");
    return V;
  }

  if (const BinaryOperator *BOp = dyn_cast<BinaryOperator>(V)) {
    if (ConstantInt *RHSC = dyn_cast<ConstantInt>(BOp->getOperand(1))) {

      // If we've been called recursively, then Offset and Scale will be wider
      // than the BOp operands. We'll always zext it here as we'll process sign
      // extensions below (see the isa<SExtInst> / isa<ZExtInst> cases).
      APInt RHS = RHSC->getValue().zextOrSelf(Offset.getBitWidth());

      switch (BOp->getOpcode()) {
      default:
        // We don't understand this instruction, so we can't decompose it any
        // further.
        Scale = 1;
        Offset = 0;
        return V;
      case Instruction::Or:
        // X|C == X+C if all the bits in C are unset in X.  Otherwise we can't
        // analyze it.
        if (!MaskedValueIsZero(BOp->getOperand(0), RHSC->getValue(), DL, 0, AC,
                               BOp, DT)) {
          Scale = 1;
          Offset = 0;
          return V;
        }
        LLVM_FALLTHROUGH;
      case Instruction::Add:
        V = GetLinearExpression(BOp->getOperand(0), Scale, Offset, ZExtBits,
                                SExtBits, DL, Depth + 1, AC, DT, NSW, NUW);
        Offset += RHS;
        break;
      case Instruction::Sub:
        V = GetLinearExpression(BOp->getOperand(0), Scale, Offset, ZExtBits,
                                SExtBits, DL, Depth + 1, AC, DT, NSW, NUW);
        Offset -= RHS;
        break;
      case Instruction::Mul:
        V = GetLinearExpression(BOp->getOperand(0), Scale, Offset, ZExtBits,
                                SExtBits, DL, Depth + 1, AC, DT, NSW, NUW);
        Offset *= RHS;
        Scale *= RHS;
        break;
      case Instruction::Shl:
        V = GetLinearExpression(BOp->getOperand(0), Scale, Offset, ZExtBits,
                                SExtBits, DL, Depth + 1, AC, DT, NSW, NUW);
        Offset <<= RHS.getLimitedValue();
        Scale <<= RHS.getLimitedValue();
        // the semantics of nsw and nuw for left shifts don't match those of
        // multiplications, so we won't propagate them.
        NSW = NUW = false;
        return V;
      }

      if (isa<OverflowingBinaryOperator>(BOp)) {
        NUW &= BOp->hasNoUnsignedWrap();
        NSW &= BOp->hasNoSignedWrap();
      }
      return V;
    }
  }

  // Since GEP indices are sign extended anyway, we don't care about the high
  // bits of a sign or zero extended value - just scales and offsets.  The
  // extensions have to be consistent though.
  if (isa<SExtInst>(V) || isa<ZExtInst>(V)) {
    Value *CastOp = cast<CastInst>(V)->getOperand(0);
    unsigned NewWidth = V->getType()->getPrimitiveSizeInBits();
    unsigned SmallWidth = CastOp->getType()->getPrimitiveSizeInBits();
    unsigned OldZExtBits = ZExtBits, OldSExtBits = SExtBits;
    const Value *Result =
        GetLinearExpression(CastOp, Scale, Offset, ZExtBits, SExtBits, DL,
                            Depth + 1, AC, DT, NSW, NUW);

    // zext(zext(%x)) == zext(%x), and similarly for sext; we'll handle this
    // by just incrementing the number of bits we've extended by.
    unsigned ExtendedBy = NewWidth - SmallWidth;

    if (isa<SExtInst>(V) && ZExtBits == 0) {
      // sext(sext(%x, a), b) == sext(%x, a + b)

      if (NSW) {
        // We haven't sign-wrapped, so it's valid to decompose sext(%x + c)
        // into sext(%x) + sext(c). We'll sext the Offset ourselves:
        unsigned OldWidth = Offset.getBitWidth();
        Offset = Offset.trunc(SmallWidth).sext(NewWidth).zextOrSelf(OldWidth);
      } else {
        // We may have signed-wrapped, so don't decompose sext(%x + c) into
        // sext(%x) + sext(c)
        Scale = 1;
        Offset = 0;
        Result = CastOp;
        ZExtBits = OldZExtBits;
        SExtBits = OldSExtBits;
      }
      SExtBits += ExtendedBy;
    } else {
      // sext(zext(%x, a), b) = zext(zext(%x, a), b) = zext(%x, a + b)

      if (!NUW) {
        // We may have unsigned-wrapped, so don't decompose zext(%x + c) into
        // zext(%x) + zext(c)
        Scale = 1;
        Offset = 0;
        Result = CastOp;
        ZExtBits = OldZExtBits;
        SExtBits = OldSExtBits;
      }
      ZExtBits += ExtendedBy;
    }

    return Result;
  }

  Scale = 1;
  Offset = 0;
  return V;
}

static int64_t adjustToPointerSize(int64_t Offset, unsigned PointerSize) {
  assert(PointerSize <= 64 && "Invalid PointerSize!");
  unsigned ShiftBits = 64 - PointerSize;
  return (int64_t)((uint64_t)Offset << ShiftBits) >> ShiftBits;
}


bool DecomposeGEPExpression(const Value *V,
       DecomposedGEP &Decomposed, const DataLayout &DL, AssumptionCache *AC,
       DominatorTree *DT) {
  // Limit recursion depth to limit compile time in crazy cases.
  unsigned MaxLookup = MaxLookupSearchDepth;
  //SearchTimes++;

  Decomposed.StructOffset = 0;
  Decomposed.OtherOffset = 0;
  Decomposed.VarIndices.clear();
  do {
    // See if this is a bitcast or GEP.
    const Operator *Op = dyn_cast<Operator>(V);
    if (!Op) {
      // The only non-operator case we can handle are GlobalAliases.
      if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(V)) {
        if (!GA->isInterposable()) {
          V = GA->getAliasee();
          continue;
        }
      }
      Decomposed.Base = V;
      return false;
    }

    if (Op->getOpcode() == Instruction::BitCast ||
        Op->getOpcode() == Instruction::AddrSpaceCast) {
      V = Op->getOperand(0);
      continue;
    }

    const GEPOperator *GEPOp = dyn_cast<GEPOperator>(Op);
    if (!GEPOp) {
      if (auto CS = ImmutableCallSite(V))
        if (const Value *RV = CS.getReturnedArgOperand()) {
          V = RV;
          continue;
        }

      // If it's not a GEP, hand it off to SimplifyInstruction to see if it
      // can come up with something. This matches what GetUnderlyingObject does.
      if (const Instruction *I = dyn_cast<Instruction>(V))
        // TODO: Get a DominatorTree and AssumptionCache and use them here
        // (these are both now available in this function, but this should be
        // updated when GetUnderlyingObject is updated). TLI should be
        // provided also.
        if (const Value *Simplified =
                SimplifyInstruction(const_cast<Instruction *>(I), DL)) {
          V = Simplified;
          continue;
        }

      Decomposed.Base = V;
      return false;
    }

    // Don't attempt to analyze GEPs over unsized objects.
    if (!GEPOp->getSourceElementType()->isSized()) {
      Decomposed.Base = V;
      return false;
    }

    unsigned AS = GEPOp->getPointerAddressSpace();
    // Walk the indices of the GEP, accumulating them into BaseOff/VarIndices.
    gep_type_iterator GTI = gep_type_begin(GEPOp);
    unsigned PointerSize = DL.getPointerSizeInBits(AS);
    // Assume all GEP operands are constants until proven otherwise.
    bool GepHasConstantOffset = true;
    for (User::const_op_iterator I = GEPOp->op_begin() + 1, E = GEPOp->op_end();
         I != E; ++I, ++GTI) {
      const Value *Index = *I;
      // Compute the (potentially symbolic) offset in bytes for this index.
      if (StructType *STy = GTI.getStructTypeOrNull()) {
        // For a struct, add the member offset.
        unsigned FieldNo = cast<ConstantInt>(Index)->getZExtValue();
        if (FieldNo == 0)
          continue;

        Decomposed.StructOffset +=
          DL.getStructLayout(STy)->getElementOffset(FieldNo);
        continue;
      }

      // For an array/pointer, add the element offset, explicitly scaled.
      if (const ConstantInt *CIdx = dyn_cast<ConstantInt>(Index)) {
        if (CIdx->isZero())
          continue;
        Decomposed.OtherOffset +=
          DL.getTypeAllocSize(GTI.getIndexedType()) * CIdx->getSExtValue();
        continue;
      }

      GepHasConstantOffset = false;

      uint64_t Scale = DL.getTypeAllocSize(GTI.getIndexedType());
      unsigned ZExtBits = 0, SExtBits = 0;

      // If the integer type is smaller than the pointer size, it is implicitly
      // sign extended to pointer size.
      unsigned Width = Index->getType()->getIntegerBitWidth();
      if (PointerSize > Width)
        SExtBits += PointerSize - Width;

      // Use GetLinearExpression to decompose the index into a C1*V+C2 form.
      APInt IndexScale(Width, 0), IndexOffset(Width, 0);
      bool NSW = true, NUW = true;
      Index = GetLinearExpression(Index, IndexScale, IndexOffset, ZExtBits,
                                  SExtBits, DL, 0, AC, DT, NSW, NUW);

      // The GEP index scale ("Scale") scales C1*V+C2, yielding (C1*V+C2)*Scale.
      // This gives us an aggregate computation of (C1*Scale)*V + C2*Scale.
      Decomposed.OtherOffset += IndexOffset.getSExtValue() * Scale;
      Scale *= IndexScale.getSExtValue();

      // If we already had an occurrence of this index variable, merge this
      // scale into it.  For example, we want to handle:
      //   A[x][x] -> x*16 + x*4 -> x*20
      // This also ensures that 'x' only appears in the index list once.
      for (unsigned i = 0, e = Decomposed.VarIndices.size(); i != e; ++i) {
        if (Decomposed.VarIndices[i].V == Index &&
            Decomposed.VarIndices[i].ZExtBits == ZExtBits &&
            Decomposed.VarIndices[i].SExtBits == SExtBits) {
          Scale += Decomposed.VarIndices[i].Scale;
          Decomposed.VarIndices.erase(Decomposed.VarIndices.begin() + i);
          break;
        }
      }

      // Make sure that we have a scale that makes sense for this target's
      // pointer size.
      Scale = adjustToPointerSize(Scale, PointerSize);

      if (Scale) {
        VariableGEPIndex Entry = {Index, ZExtBits, SExtBits,
                                  static_cast<int64_t>(Scale)};
        Decomposed.VarIndices.push_back(Entry);
      }
    }

    // Take care of wrap-arounds
    if (GepHasConstantOffset) {
      Decomposed.StructOffset =
          adjustToPointerSize(Decomposed.StructOffset, PointerSize);
      Decomposed.OtherOffset =
          adjustToPointerSize(Decomposed.OtherOffset, PointerSize);
    }

    // Analyze the base pointer next.
    V = GEPOp->getOperand(0);
  } while (--MaxLookup);

  // If the chain of expressions is too deep, just return early.
  Decomposed.Base = V;
  //SearchLimitReached++;
  return true;
}

int getInstructionID(Instruction *II) 
{
	MDNode * Node = II->getMetadata("ins_id");

	if (!Node) 
	{
		return -1;
    }

	assert(Node->getNumOperands() == 1);
	const Metadata *MD = Node->getOperand(0);
	if (auto *MDV = dyn_cast<ValueAsMetadata>(MD)) 
	{
		Value *V = MDV->getValue();
		ConstantInt *CI = dyn_cast<ConstantInt>(V);
		assert(CI);
		return CI->getZExtValue();
	}

	return -1; 
}

char Debloater::ID = 0;

void Debloater::getAnalysisUsage(AnalysisUsage &AU) const {
	AU.setPreservesCFG();
	AU.addRequired<PostDominatorTreeWrapperPass>();
	AU.addRequired<AssumptionCacheTracker>();
	AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
}

Debloater::Debloater() : ModulePass(ID) 
{
	PassRegistry &Registry = *PassRegistry::getPassRegistry();
	initializePostDominatorTreeWrapperPassPass(Registry);
	initializeDominatorTreeWrapperPassPass(Registry);
    initializeLoopInfoWrapperPassPass(Registry);
   
}

bool Debloater::isTargetInstruction(Instruction * pInst)
{
	if(LoadInst * pLoad = dyn_cast<LoadInst>(pInst))
	{
		Value * pPointerValue = pLoad->getPointerOperand();
		Value * pBase = GetUnderlyingObject(pPointerValue, *this->dl);

		if(PointerType * pPType = dyn_cast<PointerType>(pBase->getType()))
		{
			if(StructType * pStType = dyn_cast<StructType>(pPType->getElementType()))
			{
				if(pStType->getName() == "struct.MIDI_MSG")
				{
					return true;
				}
			}
		}
	}
	else if(StoreInst * pStore =dyn_cast<StoreInst>(pInst))
	{
		Value * pPointerValue = pStore->getPointerOperand();
		Value * pBase = GetUnderlyingObject(pPointerValue, *this->dl);

		if(PointerType * pPType = dyn_cast<PointerType>(pBase->getType()))
		{
			if(StructType * pStType = dyn_cast<StructType>(pPType->getElementType()))
			{
				if(pStType->getName() == "struct.MIDI_MSG")
				{
					return true;
				}
			}
		}
	}

	return false;
}

void Debloater::removeInstruction(Instruction * pInst)
{

	vector<Instruction *> vecWorkList;
	vecWorkList.push_back(pInst);

	while(vecWorkList.size() > 0)
	{
		Instruction * I = vecWorkList.back();
		vecWorkList.pop_back();

		if(I->getNumUses() == 0)
		{
			for(unsigned i = 0; i < I->getNumOperands(); i ++)
			{
				if(Instruction * pI = dyn_cast<Instruction>(I->getOperand(i)))
				{
					vecWorkList.push_back(pI);
				}
			}

			this->numTotalRemove++;
			I->eraseFromParent();
		}
	}
}


void Debloater::removeUnnecessaryWrite(map<int64_t, vector<LoadInst *> > & mapOffsetRead, map<int64_t, vector<StoreInst *> > mapOffsetWrite)
{
	map<int64_t, vector<StoreInst *> >::iterator itMapBegin; 
	//map<int64_t, vector<StoreInst *> >::iterator itMapEnd   = mapOffsetWrite.end();

	for(itMapBegin = mapOffsetWrite.begin(); itMapBegin != mapOffsetWrite.end(); itMapBegin ++ )
	{

    /*
		for(unsigned i = 0; i < itMapBegin->second.size(); i ++)
		{
			if(getInstructionID(itMapBegin->second[i]) == 3101)
			{
				errs() << itMapBegin->first << "\n";

				for(unsigned j = 0; j < mapOffsetRead[itMapBegin->first].size(); j ++ )
				{
					mapOffsetRead[itMapBegin->first][j]->dump();
					if(MDNode * N = mapOffsetRead[itMapBegin->first][j]->getMetadata("dbg"))
					{
						const DILocation *Loc = mapOffsetRead[itMapBegin->first][j]->getDebugLoc();
					
						errs() << "//-- " << getInstructionID(mapOffsetRead[itMapBegin->first][j]) << " "  << Loc->getFilename() << ": " << Loc->getLine() << "\n";
					}
					//unsigned int uLineNoForInst = Loc->getLine();
				}
			}
		}
    */
		
		if(mapOffsetRead.find(itMapBegin->first) == mapOffsetRead.end())
		{
			for(unsigned i = 0; i < itMapBegin->second.size(); i ++)
			{
				StoreInst * pStore = itMapBegin->second[i];
				if(MDNode * N = pStore->getMetadata("dbg"))
				{
					const DILocation *Loc = pStore->getDebugLoc();
					pStore->dump();
					errs() << "//-- " << getInstructionID(pStore) << " "  << Loc->getFilename() << ": " << Loc->getLine() << "\n";
					//unsigned int uLineNoForInst = Loc->getLine();

				  removeInstruction(pStore);

				}
			}
		}

	}
}

/*
void Debloater::collectLoopDependence(Function * pFunction, map<LoadInst *, int64_t> & mapReadOffset, map<LoadInst *, map<int64_t, vector<int64_t> > > & mapUseDependence)
{
	PostDominatorTree *PDT = &getAnalysis<PostDominatorTreeWrapperPass>(*pFunction).getPostDomTree();

	ControlDependenceGraphBase CDG;
	CDG.graphForFunction(*pFunction, *PDT);

	LoopInfo * pLI = &getAnalysis<LoopInfoWrapperPass>(*pFunction).getLoopInfo();
	Loop * pLoop = searchLoopByLineNo(pFunction, pLI, uLoopSrcLine);

	vector<BasicBlock *> vecLoopBBs;

	for(Loop::block_iterator BB = pLoop->block_begin(); BB != pLoop->block_end(); BB ++)
	{
		vecLoopBBs.push_back(*BB);
	}

	//map<LoadInst *, map<int64_t, vector<int64_t> > > mapUseDependence;

	for(Loop::block_iterator BB = pLoop->block_begin(); BB != pLoop->block_end(); BB ++ )
	{
		for(BasicBlock::iterator II = (*BB)->begin(); II != (*BB)->end(); II ++)
		{
			if(LoadInst * pLoad = dyn_cast<LoadInst>(II))
			{
				if(isTargetInstruction(pLoad))
				{
					BasicBlock * pBB = pLoad->getParent();

					for(unsigned i = 0; i < vecLoopBBs.size(); i ++)
					{
						if(pBB == vecLoopBBs[i])
						{
							continue;
						}

						if(CDG.influences(vecLoopBBs[i], pBB))
						{
							if(SwitchInst * pSwitch = dyn_cast<SwitchInst>(vecLoopBBs[i]->getTerminator()))
							{	
								if(LoadInst * pConLoad = dyn_cast<LoadInst>(pSwitch->getCondition()))
								{
									if(isTargetInstruction(pConLoad))
									{
										map<int64_t, vector<int64_t> > mapTmp;

										//errs() << mapReadOffset[pConLoad] << "\n";

										for(SwitchInst::CaseIt it = pSwitch->case_begin(); it != pSwitch->case_end(); it ++ )
										{
											if(ConstantInt * pConst = dyn_cast<ConstantInt>(it->getCaseValue()))
											{
												mapUseDependence[pLoad][mapReadOffset[pConLoad]].push_back(pConst->getSExtValue());
											}	
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}


	map<LoadInst *, map<int64_t, vector<int64_t> > >::iterator itUseMapBegin = mapUseDependence.begin();
	map<LoadInst *, map<int64_t, vector<int64_t> > >::iterator itUseMapEnd = mapUseDependence.end();

	while(itUseMapBegin != itUseMapEnd)
	{
		itUseMapBegin->first->dump();
		//errs() << itUseMapBegin->second.size() << "\n";

		map<int64_t, vector<int64_t> >::iterator itMapBegin = itUseMapBegin->second.begin();
		map<int64_t, vector<int64_t> >::iterator itMapEnd = itUseMapBegin->second.end();

		while(itMapBegin != itMapEnd)
		{
			errs() << itMapBegin->first << ": ";

			for(unsigned i = 0; i < itMapBegin->second.size(); i ++ )
			{
				errs() << itMapBegin->second[i] << " ";
			}

			errs() << "\n";

			itMapBegin ++;
		}

		itUseMapBegin++;
	}

}

*/
/*
void Debloater::removeConditionalDef(Function * pFunction, map<LoadInst *, int64_t> & mapReadOffset)
{
	
}



void Debloater::removeByControlDep(map<int64_t, vector<LoadInst *> > & mapOffsetRead, map<int64_t, vector<StoreInst *> > mapOffsetWrite)
{
	map<int64_t, vector<LoadInst *> >::iterator itMapBegin = mapOffsetRead.begin();
	map<int64_t, vector<LoadInst *> >::iterator itMapEnd   = mapOffsetRead.end();

	for(; itMapBegin != itMapEnd; itMapBegin ++ )
	{
		for(unsigned i = 0; i < itMapBegin->second.size(); i ++ )
		{

		}
	}




	for(unsigned i = 0; i < mapOffsetRead[32].size(); i ++ )
	{
		mapOffsetRead[32][i]->dump();

		if(MDNode * N = mapOffsetRead[32][i]->getMetadata("dbg"))
		{
			const DILocation *Loc = mapOffsetRead[32][i]->getDebugLoc();
			errs() << "//-- " << getInstructionID(mapOffsetRead[32][i]) << " "  << Loc->getFilename() << ": " << Loc->getLine() << "\n";
		}
				//{
				//	const DILocation *Loc = pStore->getDebugLoc();
					//pStore->dump();
					//errs() << "//-- " << getInstructionID(pStore) << " "  << Loc->getFilename() << ": " << Loc->getLine() << "\n";
					//unsigned int uLineNoForInst = Loc->getLine();

	}


	map<int64_t, vector<LoadInst *> >::iterator itLoadMapBegin;

	for(itLoadMapBegin = mapOffsetRead.begin(); itLoadMapBegin != mapOffsetRead.end(); itLoadMapBegin ++ )
	{
		set<Function *> setParentFunction;

		for(unsigned i = 0; i < itLoadMapBegin->second.size(); i ++ )
		{
			setParentFunction.insert(itLoadMapBegin->second[i]->getParent()->getParent());
		}

		if(setParentFunction.size() == 1)
		{
			if((*(setParentFunction.begin()))->getName() == "ConvertMIDI")
			{
				for(unsigned i = 0; i < itLoadMapBegin->second.size(); i ++ )
				{
					itLoadMapBegin->second[i]->dump();
				}
			}
		}
	}

}

*/

bool Debloater::runOnModule(Module & M)
{
	Function * pMain = M.getFunction("main");

	Function * pFunction = M.getFunction(strFuncName);

	if(pFunction == NULL)
	{
		errs() << "Cannot find the input function!\n" ;
		return false;
	}

	DominatorTree * DT = &(getAnalysis<DominatorTreeWrapperPass>(*pFunction).getDomTree());
	//PostDominatorTree *PDT = &getAnalysis<PostDominatorTreeWrapperPass>(*pFunction).getPostDomTree();
	LoopInfo * pLI = &getAnalysis<LoopInfoWrapperPass>(*pFunction).getLoopInfo();
	AssumptionCache * AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(*pFunction);

	Loop * pLoop = searchLoopByLineNo(pFunction, pLI, uLoopSrcLine);

	
	
	this->dl = new DataLayout(&M);

	map<LoadInst *, int64_t> mapReadOffset;
	map<int64_t, vector<LoadInst *> > mapOffsetRead;

	map<StoreInst *, int64_t> mapWriteOffset;
	map<int64_t, vector<StoreInst *> > mapOffsetWrite;

	vector<Function *> vecWorkList;

	for(Loop::block_iterator BB = pLoop->block_begin(); BB != pLoop->block_end(); BB ++ )
	{
		for(BasicBlock::iterator II = (*BB)->begin(); II != (*BB)->end(); II ++)
		{
			if(LoadInst * pLoad = dyn_cast<LoadInst>(II))
			{
				if(isTargetInstruction(pLoad))
				{
					DecomposedGEP de;
					DecomposeGEPExpression(pLoad->getPointerOperand(), de, *this->dl, AC, DT);

					mapReadOffset[pLoad] = de.StructOffset;
					mapOffsetRead[de.StructOffset].push_back(pLoad);
				}
			}
			else if(StoreInst * pStore = dyn_cast<StoreInst>(II))
			{
				if(isTargetInstruction(pStore))
				{
					DecomposedGEP de;
					DecomposeGEPExpression(pStore->getPointerOperand(), de, *this->dl, AC, DT);

					mapWriteOffset[pStore] = de.StructOffset;
					mapOffsetWrite[de.StructOffset].push_back(pStore);
				}
			}
			else if(CallInst * pCall = dyn_cast<CallInst>(II))
			{
				if(pCall->getCalledFunction() != NULL)
				{
					vecWorkList.push_back(pCall->getCalledFunction());
				}

			}
		}
	}

	set<Function *> setProcessedFunc;

	while(vecWorkList.size() > 0)
	{
		Function * pF = vecWorkList.back();
		vecWorkList.pop_back();

		if(setProcessedFunc.find(pF) != setProcessedFunc.end() )
		{
			continue;
		}

		if(pF->begin() == pF->end())
		{
			continue;
		}

		DT = &(getAnalysis<DominatorTreeWrapperPass>(*pF).getDomTree());

		setProcessedFunc.insert(pF);

		for(Function::iterator BB = pF->begin(); BB != pF->end(); BB ++ )
		{
			for(BasicBlock::iterator II = BB->begin(); II != BB->end(); II ++ )
			{
				if(LoadInst * pLoad = dyn_cast<LoadInst>(II))
				{
					if(isTargetInstruction(pLoad))
					{
						DecomposedGEP de;
						DecomposeGEPExpression(pLoad->getPointerOperand(), de, *this->dl, AC, DT);

						mapReadOffset[pLoad] = de.StructOffset;
						mapOffsetRead[de.StructOffset].push_back(pLoad);
					}
				}
				else if(StoreInst * pStore = dyn_cast<StoreInst>(II))
				{
					if(isTargetInstruction(pStore))
					{
						DecomposedGEP de;
						DecomposeGEPExpression(pStore->getPointerOperand(), de, *this->dl, AC, DT);

						mapWriteOffset[pStore] = de.StructOffset;
						mapOffsetWrite[de.StructOffset].push_back(pStore);
					}
				}
				else if(CallInst * pCall = dyn_cast<CallInst>(II))
				{
					if(pCall->getCalledFunction() != NULL)
					{
						vecWorkList.push_back(pCall->getCalledFunction());
					}

				}
			}
		}
	}

  removeUnnecessaryWrite(mapOffsetRead, mapOffsetWrite);

  errs() << this->numTotalRemove << "\n";

  Function * pLibrary = M.getFunction("midiReadGetNextMessage");

  int numCount = 0;

  for(Function::iterator BB = pLibrary->begin(); BB != pLibrary->end(); BB ++)
  {
    for(BasicBlock::iterator II = BB->begin(); II != BB->end(); II ++)
    {
      numCount++;
    }
  }

  errs() << numCount << "\n";
	//map<LoadInst *, map<int64_t, vector<int64_t> > > mapUseDependence;

	//collectLoopDependence(pFunction, mapReadOffset, mapUseDependence);

	return true;
}
