package us.ihmc.convexOptimization.quadraticProgram;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;

public interface InverseCostCalculator<T extends DMatrix>
{
   void computeInverse(T matrix, T inverseMatrix);
}