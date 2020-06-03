package us.ihmc.convexOptimization.quadraticProgram;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.factory.LinearSolverFactory_DDRM;
import org.ejml.interfaces.linsol.LinearSolverDense;

import us.ihmc.matrixlib.MatrixTools;

// Uses the algorithm and naming convention found in MIT Paper
// "An efficiently solvable quadratic program for stabilizing dynamic locomotion"
// by Scott Kuindersma, Frank Permenter, and Russ Tedrake.
public class SimpleActiveSetQPStandaloneSolver
{
   DMatrixRMaj wInverse = new DMatrixRMaj(0);
   DMatrixRMaj minusWInverseG = new DMatrixRMaj(0);
   DMatrixRMaj gVector = new DMatrixRMaj(0);
   DMatrixRMaj alphaAndGamma = new DMatrixRMaj(0);
   DMatrixRMaj rMatrix = new DMatrixRMaj(0);

   DMatrixRMaj wInverseRTranspose = new DMatrixRMaj(0);
   DMatrixRMaj leftSide = new DMatrixRMaj(0);
   DMatrixRMaj rightSide = new DMatrixRMaj(0);
   DMatrixRMaj eVector = new DMatrixRMaj(0);
   LinearSolverDense<DMatrixRMaj> linearSolver = LinearSolverFactory_DDRM.linear(0);

   int maxIterations;

   public SimpleActiveSetQPStandaloneSolver()
   {
      this(1000);
   }

   public SimpleActiveSetQPStandaloneSolver(int maxIterations)
   {
      this.maxIterations = maxIterations;
   }

   public int solve(DMatrixRMaj quadraticCostGMatrix, DMatrixRMaj quadraticCostFVector, DMatrixRMaj linearEqualityConstraintA,
                    DMatrixRMaj linearEqualityConstraintB, DMatrixRMaj linearInequalityConstraintA, DMatrixRMaj linearInequalityConstraintB,
                    boolean[] linearInequalityActiveSet, DMatrixRMaj solutionVector)
   {
      int numberOfVariablesToSolve = quadraticCostGMatrix.numCols;
      int iterations = 0;

      // precomputed constants over iterations
      if (quadraticCostGMatrix instanceof BlockDiagSquareMatrix)
      {
         if (!(wInverse instanceof BlockDiagSquareMatrix))
         {
            wInverse = new BlockDiagSquareMatrix(((BlockDiagSquareMatrix) quadraticCostGMatrix).blockSizes);
         }
         ((BlockDiagSquareMatrix) quadraticCostGMatrix).packInverse(linearSolver, (BlockDiagSquareMatrix) wInverse);

      }
      else
      {
         //CommonOps.invert(quadraticCostGMatrix, wInverse);    //this is quite slow
         wInverse.reshape(quadraticCostGMatrix.numRows, quadraticCostGMatrix.numCols);
         linearSolver.setA(quadraticCostGMatrix);
         linearSolver.invert(wInverse);
      }

      gVector.reshape(quadraticCostFVector.numRows, quadraticCostFVector.numCols);
      gVector.set(quadraticCostFVector);

      minusWInverseG.reshape(wInverse.numRows, 1);
      if (wInverse instanceof BlockDiagSquareMatrix)
         ((BlockDiagSquareMatrix) wInverse).mult(-1, gVector, minusWInverseG);
      else
         CommonOps_DDRM.mult(-1, wInverse, gVector, minusWInverseG);

      int linearEqualityConstraintsSize = 0;
      if (linearEqualityConstraintA != null)
      {
         linearEqualityConstraintsSize = linearEqualityConstraintA.numRows;
         rMatrix.reshape(linearEqualityConstraintsSize, numberOfVariablesToSolve);
         CommonOps_DDRM.insert(linearEqualityConstraintA, rMatrix, 0, 0);
         eVector.reshape(linearEqualityConstraintsSize, 1);
         CommonOps_DDRM.insert(linearEqualityConstraintB, eVector, 0, 0);
      }

      solutionVector.reshape(numberOfVariablesToSolve, 1);

      boolean done = false;

      while (!done)
      {
         int activeInequalityConstraintSize = getNumOfTrue(linearInequalityActiveSet);
         int activeConstraintSize = linearEqualityConstraintsSize + activeInequalityConstraintSize;
         alphaAndGamma.reshape(activeConstraintSize, 1);

         if (linearEqualityConstraintsSize > 0 || activeInequalityConstraintSize > 0) // if there is any constraints
         {
            // rMatrix = [linearEqualityConstraintMatrix; activeLinearInequalityConstraintMatrix];
            rMatrix.reshape(activeConstraintSize, numberOfVariablesToSolve, true);
            if (activeInequalityConstraintSize > 0)
               setPartialMatrixForInequalityConstraints(linearInequalityConstraintA, linearInequalityActiveSet, linearEqualityConstraintsSize, 0, rMatrix);

            // eVector = [linearEqualityConstraintVector; activeLinearInequalityConstraintVector];
            eVector.reshape(activeConstraintSize, 1, true);
            if (activeInequalityConstraintSize > 0)
               setPartialVectorForInequalityConstraints(linearInequalityConstraintB, linearInequalityActiveSet, linearEqualityConstraintsSize, eVector);

            // wInverse * R'
            wInverseRTranspose.reshape(numberOfVariablesToSolve, activeConstraintSize);
            if (wInverse instanceof BlockDiagSquareMatrix)
            {
               ((BlockDiagSquareMatrix) wInverse).multTransB(rMatrix, wInverseRTranspose);
            }
            else
            {
               CommonOps_DDRM.multTransB(wInverse, rMatrix, wInverseRTranspose);
            }

            // LHS
            leftSide.reshape(activeConstraintSize, activeConstraintSize);
            CommonOps_DDRM.mult(-1, rMatrix, wInverseRTranspose, leftSide);

            // RHS
            rightSide.reshape(activeConstraintSize, 1);
            rightSide.set(eVector);
            CommonOps_DDRM.multAddTransA(wInverseRTranspose, gVector, rightSide);
            linearSolver.setA(leftSide);
            linearSolver.solve(rightSide, alphaAndGamma);

            // solve z from alphaAndGamma
            solutionVector.set(minusWInverseG);
            CommonOps_DDRM.multAdd(-1, wInverseRTranspose, alphaAndGamma, solutionVector);
         }

         else
         {
            CommonOps_DDRM.mult(-1, wInverse, gVector, solutionVector);
         }

         iterations++;

         if (iterations > maxIterations)
         {
            done = true;
         }
         else if (linearInequalityActiveSet != null)
         {
            done = true;

            int activeLinearInequalityCount = 0;
            for (int i = 0; i < linearInequalityActiveSet.length; i++)
            {
               if (linearInequalityActiveSet[i] == false)
               {
                  // For each element not in the active set, check to see
                  // if it should be in the active set if p_i^T z > f_i:

                  double pz = MatrixTools.multMatrixRowVector(linearInequalityConstraintA, i, solutionVector);
                  if (pz > linearInequalityConstraintB.get(i, 0))
                  {
                     linearInequalityActiveSet[i] = true;
                     done = false;
                  }
               }
               else
               {
                  // For each element in the active set, check to see if
                  // it should be taken out of the active set if gamma_i <
                  // 0.0:

                  double gamma = alphaAndGamma.get(linearEqualityConstraintsSize + activeLinearInequalityCount++, 0);
                  if (gamma < 0.0)
                  {
                     linearInequalityActiveSet[i] = false;
                     done = false;
                  }
               }
            }
         }
      }

      if (iterations > maxIterations)
         return -1;
      else
         return iterations;
   }

   private int getNumOfTrue(boolean[] linearInequalityActiveSet)
   {
      if (linearInequalityActiveSet == null)
         return 0;
      int count = 0;
      for (int i = 0; i < linearInequalityActiveSet.length; i++)
      {
         count += linearInequalityActiveSet[i] ? 1 : 0;
      }

      return count;
   }

   protected static void setPartialMatrixForInequalityConstraints(DMatrixRMaj fromMatrix, boolean[] isActiveRowInMatrix, int startRow, int startColumn,
                                                                  DMatrixRMaj toMatrix)
   {
      int activeRow = 0;

      for (int i = 0; i < fromMatrix.numRows; i++)
      {
         if (isActiveRowInMatrix[i])
         {
            CommonOps_DDRM.extract(fromMatrix, i, i + 1, 0, fromMatrix.numCols, toMatrix, startRow + activeRow, startColumn);

            //          for (int j = 0; j < fromMatrix.numCols; j++)
            //          {
            //             toMatrix.set(startRow + activeRow, startColumn + j, fromMatrix.get(i, j));
            //          }
            activeRow++;
         }
      }

   }

   protected void setPartialVectorForInequalityConstraints(DMatrixRMaj fromVector, boolean[] isActiveRowInVector, int startRow, DMatrixRMaj toVector)
   {
      int activeRow = 0;

      for (int i = 0; i < fromVector.numRows; i++)
      {
         if (isActiveRowInVector[i])
         {
            toVector.set(startRow + activeRow, 0, fromVector.get(i, 0));

            activeRow++;
         }
      }
   }

}