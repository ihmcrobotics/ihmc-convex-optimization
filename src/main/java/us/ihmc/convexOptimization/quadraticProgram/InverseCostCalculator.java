package us.ihmc.convexOptimization.quadraticProgram;

import org.ejml.data.DMatrix;

public interface InverseCostCalculator<T extends DMatrix>
{
   void computeInverse(T matrix, T inverseMatrix);
}
