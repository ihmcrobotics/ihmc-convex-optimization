package us.ihmc.convexOptimization.quadraticProgram;

import org.ejml.data.DMatrixRMaj;

import us.ihmc.convexOptimization.exceptions.NoConvergenceException;

public abstract class ConstrainedQPSolver
{

   /*
    * minimizex (1/2)x'Qx+f'x s.t. Ain x <= bin Aeq x = beq,
    */
   public abstract int solve(DMatrixRMaj Q, DMatrixRMaj f, DMatrixRMaj Aeq, DMatrixRMaj beq, DMatrixRMaj Ain, DMatrixRMaj bin,
                             DMatrixRMaj x, boolean initialize)
         throws NoConvergenceException;

   public abstract int solve(DMatrixRMaj Q, DMatrixRMaj f, DMatrixRMaj Aeq, DMatrixRMaj beq, DMatrixRMaj Ain, DMatrixRMaj bin,
                             DMatrixRMaj lb, DMatrixRMaj ub, DMatrixRMaj x, boolean initialize)
         throws NoConvergenceException;

   public abstract boolean supportBoxConstraints();

   static double[][] DenseMatrixToDoubleArray(DMatrixRMaj Q)
   {
      double[][] Qarray = new double[Q.numRows][Q.numCols];
      for (int i = 0; i < Q.numRows; i++)
         System.arraycopy(Q.getData(), Q.numCols * i, Qarray[i], 0, Q.numCols);
      return Qarray;
   }

}
