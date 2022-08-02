package us.ihmc.convexOptimization.linearProgram;

import org.ejml.data.DMatrixRMaj;
import org.ejml.ops.MatrixIO;
import us.ihmc.euclid.tools.EuclidCoreIOTools;
import us.ihmc.matrixlib.MatrixTools;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.Arrays;

import static us.ihmc.convexOptimization.linearProgram.DictionaryFormLinearProgramSolver.maxVariables;

public class LinearProgramSolver
{
   private final DMatrixRMaj startingDictionary = new DMatrixRMaj(maxVariables, maxVariables);
   private final DictionaryFormLinearProgramSolver dictionaryFormSolver = new DictionaryFormLinearProgramSolver();
   private DMatrixRMaj augmentedInequalityMatrix = new DMatrixRMaj(0);
   private DMatrixRMaj augmentedInequalityVector = new DMatrixRMaj(0);

   /**
    * Solves a standard form linear program
    *
    * <p>
    * max c<sup>T</sup>x, Ax <= b, x >= 0.
    * </p>
    * <p>
    * Returns true if an optimal solution is computed or false otherwise.
    */
   public boolean solve(DMatrixRMaj costVectorC, DMatrixRMaj inequalityConstraintMatrixA, DMatrixRMaj inequalityConstraintVectorB, DMatrixRMaj solutionToPack)
   {
      return solve(costVectorC, inequalityConstraintMatrixA, inequalityConstraintVectorB, solutionToPack, SolverMethod.SIMPLEX);
   }

   /**
    * Solves a standard form linear program
    * with Equality Constraints
    *
    * <p>
    * max c<sup>T</sup>x, Ax <= b, Cx == d, x >= 0.
    * </p>
    * <p>
    * Returns true if an optimal solution is computed or false otherwise.
    */
   public boolean solve(DMatrixRMaj costVectorC, DMatrixRMaj inequalityConstraintMatrixA, DMatrixRMaj inequalityConstraintVectorB,
                        DMatrixRMaj equalityConstraintMatrixC, DMatrixRMaj equalityConstraintVectorD, DMatrixRMaj solutionToPack, SolverMethod solverMethod)
   {
      if (costVectorC.getNumCols() != 1 || inequalityConstraintVectorB.getNumCols() != 1 || equalityConstraintVectorD.getNumCols() != 1)
         throw new IllegalArgumentException("Invalid matrix dimensions.");
      if (inequalityConstraintMatrixA.getNumCols() != costVectorC.getNumRows())
         throw new IllegalArgumentException("Invalid matrix dimensions.");
      if ( inequalityConstraintMatrixA.getNumRows() != inequalityConstraintVectorB.getNumRows() )
         throw new IllegalArgumentException("Invalid matrix dimensions.");
      if ( equalityConstraintMatrixC.getNumRows() != equalityConstraintVectorD.getNumRows() )
         throw new IllegalArgumentException("Invalid matrix dimensions.");
      if( inequalityConstraintMatrixA.getNumCols() != equalityConstraintMatrixC.getNumCols() )
         throw new IllegalArgumentException("Invalid matrix dimensions.");

      /* Pack new Matrices A' and b' */
      int constraints = inequalityConstraintMatrixA.getNumRows() + (2 * equalityConstraintMatrixC.getNumRows());
      int dimensionality = inequalityConstraintMatrixA.getNumCols();
      DMatrixRMaj A = new DMatrixRMaj(constraints, dimensionality);

      MatrixTools.setMatrixBlock(A, 0, 0, inequalityConstraintMatrixA, 0, 0, inequalityConstraintMatrixA.getNumRows(), inequalityConstraintMatrixA.getNumCols(), 1.0);
      MatrixTools.setMatrixBlock(A, inequalityConstraintMatrixA.getNumRows(), 0, equalityConstraintMatrixC, 0, 0, equalityConstraintMatrixC.getNumRows(), equalityConstraintMatrixC.getNumCols(), 1.0);
      MatrixTools.setMatrixBlock(A, inequalityConstraintMatrixA.getNumRows() + equalityConstraintMatrixC.getNumRows(), 0, equalityConstraintMatrixC, 0, 0, equalityConstraintMatrixC.getNumRows(), equalityConstraintMatrixC.getNumCols(), -1.0);
      augmentedInequalityMatrix = A;

      DMatrixRMaj b = new DMatrixRMaj(constraints, 1);
      MatrixTools.setMatrixBlock(b, 0, 0, inequalityConstraintVectorB, 0, 0, inequalityConstraintVectorB.getNumRows(), inequalityConstraintVectorB.getNumCols(), 1.0);
      MatrixTools.setMatrixBlock(b, inequalityConstraintVectorB.getNumRows(), 0, equalityConstraintVectorD, 0, 0, equalityConstraintVectorD.getNumRows(), equalityConstraintVectorD.getNumCols(), 1.0);
      MatrixTools.setMatrixBlock(b, inequalityConstraintVectorB.getNumRows() + equalityConstraintVectorD.getNumRows(), 0, equalityConstraintVectorD, 0, 0, equalityConstraintVectorD.getNumRows(), equalityConstraintVectorD.getNumCols(), -1.0);
      augmentedInequalityVector = b;

      return solve(costVectorC, augmentedInequalityMatrix, augmentedInequalityVector, solutionToPack, solverMethod);
   }

   /**
    * Solves a standard form linear program
    * Using the selected SolverMethod [Simplex, or CrissCross]
    * Defaults to solving Simplex if an invalid Method is supplied/
    *
    * <p>
    * max c<sup>T</sup>x, Ax <= b, x >= 0.
    * </p>
    * <p>
    * Returns true if an optimal solution is computed or false otherwise.
    */
   public boolean solve(DMatrixRMaj costVectorC, DMatrixRMaj inequalityConstraintMatrixA, DMatrixRMaj inequalityConstraintVectorB, DMatrixRMaj solutionToPack, SolverMethod solverMethod)
   {
      if (costVectorC.getNumCols() != 1 || inequalityConstraintVectorB.getNumCols() != 1)
         throw new IllegalArgumentException("Invalid matrix dimensions.");
      if (inequalityConstraintMatrixA.getNumCols() != costVectorC.getNumRows())
         throw new IllegalArgumentException("Invalid matrix dimensions.");
      if (inequalityConstraintMatrixA.getNumRows() != inequalityConstraintVectorB.getNumRows())
         throw new IllegalArgumentException("Invalid matrix dimensions.");

      startingDictionary.reshape(1 + inequalityConstraintMatrixA.getNumRows(), 1 + inequalityConstraintMatrixA.getNumCols());
      Arrays.fill(startingDictionary.getData(), 0.0);

      MatrixTools.setMatrixBlock(startingDictionary, 1, 0, inequalityConstraintVectorB, 0, 0, inequalityConstraintVectorB.getNumRows(), 1, 1.0);
      MatrixTools.setMatrixBlock(startingDictionary, 1, 1, inequalityConstraintMatrixA, 0, 0, inequalityConstraintMatrixA.getNumRows(), inequalityConstraintMatrixA.getNumCols(), -1.0);

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
