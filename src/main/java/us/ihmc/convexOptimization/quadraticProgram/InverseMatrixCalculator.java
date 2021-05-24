package us.ihmc.convexOptimization.quadraticProgram;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;

public interface InverseMatrixCalculator<T extends DMatrix>
{
   void computeInverse(T matrixToInvert, T inverseMatrixToPack);
}