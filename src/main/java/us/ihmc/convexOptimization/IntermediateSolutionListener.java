package us.ihmc.convexOptimization;

import gnu.trove.list.array.TIntArrayList;
import us.ihmc.matrixlib.NativeMatrix;

public interface IntermediateSolutionListener
{
   void reportSolution(NativeMatrix xSolutionToPack,
                       TIntArrayList activeInequalityConstraints,
                       TIntArrayList activeLowerBoundConstraints,
                       TIntArrayList activeUpperBoundConstraints);
}
