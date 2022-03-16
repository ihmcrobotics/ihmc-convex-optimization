package us.ihmc.convexOptimization.linearProgram;

import org.ejml.data.DMatrixRMaj;
import us.ihmc.matrixlib.MatrixTools;

import java.util.Arrays;

import static us.ihmc.convexOptimization.linearProgram.DictionaryFormLinearProgramSolver.maxVariables;

public class LinearProgramSolver
{
   private final DMatrixRMaj startingDictionary = new DMatrixRMaj(maxVariables, maxVariables);
   private final DictionaryFormLinearProgramSolver dictionaryFormSolver = new DictionaryFormLinearProgramSolver();

   /**
    * Solves a standard form linear program
    *
    * <p>
    * max c<sup>T</sup>x, Ax <= b, x >= 0.
    * </p>
    * <p>
    * Returns true if an optimal solution is computed or false otherwise.
    */
   public boolean solve(DMatrixRMaj costVectorC, DMatrixRMaj constraintMatrixA, DMatrixRMaj constraintVectorB, DMatrixRMaj solutionToPack)
   {
      return solve(costVectorC, constraintMatrixA, constraintVectorB, solutionToPack, SolverMethod.SIMPLEX);
   }

   public boolean solve(DMatrixRMaj costVectorC, DMatrixRMaj constraintMatrixA, DMatrixRMaj constraintVectorB, DMatrixRMaj solutionToPack, SolverMethod solverMethod)
   {
      if (costVectorC.getNumCols() != 1 || constraintVectorB.getNumCols() != 1)
         throw new IllegalArgumentException("Invalid matrix dimensions.");
      if (constraintMatrixA.getNumCols() != costVectorC.getNumRows())
         throw new IllegalArgumentException("Invalid matrix dimensions.");
      if (constraintMatrixA.getNumRows() != constraintVectorB.getNumRows())
         throw new IllegalArgumentException("Invalid matrix dimensions.");

      startingDictionary.reshape(1 + constraintMatrixA.getNumRows(), 1 + constraintMatrixA.getNumCols());
      Arrays.fill(startingDictionary.getData(), 0.0);

      MatrixTools.setMatrixBlock(startingDictionary, 1, 0, constraintVectorB, 0, 0, constraintVectorB.getNumRows(), 1, 1.0);
      MatrixTools.setMatrixBlock(startingDictionary, 1, 1, constraintMatrixA, 0, 0, constraintMatrixA.getNumRows(), constraintMatrixA.getNumCols(), -1.0);

      for (int i = 0; i < costVectorC.getNumRows(); i++)
      {
         startingDictionary.set(0, 1 + i, costVectorC.get(i));
      }

      if (solverMethod == SolverMethod.CRISS_CROSS)
      {
         dictionaryFormSolver.solveCrissCross(startingDictionary);
         if (!dictionaryFormSolver.getCrissCrossStatistics().foundSolution())
         {
            return false;
         }
      }
      else
      {
         dictionaryFormSolver.solveSimplex(startingDictionary);
         if (!dictionaryFormSolver.getPhase2Statistics().foundSolution())
         {
            return false;
         }
      }

      solutionToPack.set(dictionaryFormSolver.getSolution());
      return true;
   }
}
