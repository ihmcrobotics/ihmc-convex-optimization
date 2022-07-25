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
   public boolean solve(DMatrixRMaj costVectorC, DMatrixRMaj constraintMatrixA, DMatrixRMaj constraintVectorB, DMatrixRMaj solutionToPack)
   {
      return solve(costVectorC, constraintMatrixA, constraintVectorB, solutionToPack, SolverMethod.SIMPLEX);
   }

   public boolean solve(DMatrixRMaj costVectorC, DMatrixRMaj constraintMatrixA, DMatrixRMaj constraintVectorB,
                        DMatrixRMaj constraintMatrixC, DMatrixRMaj constraintVectorD, DMatrixRMaj solutionToPack)
   {
      if (costVectorC.getNumCols() != 1 || constraintVectorB.getNumCols() != 1 || constraintVectorD.getNumCols() != 1)
         throw new IllegalArgumentException("Invalid matrix dimensions.");
      if (constraintMatrixA.getNumCols() != costVectorC.getNumRows())
         throw new IllegalArgumentException("Invalid matrix dimensions.");
      if ( constraintMatrixA.getNumRows() != constraintVectorB.getNumRows() )
         throw new IllegalArgumentException("Invalid matrix dimensions.");
      if ( constraintMatrixC.getNumRows() != constraintVectorD.getNumRows() )
         throw new IllegalArgumentException("Invalid matrix dimensions.");
      if( constraintMatrixA.getNumCols() != constraintMatrixC.getNumCols() )
         throw new IllegalArgumentException("Invalid matrix dimensions.");

      /**  Pack new Matrices A' and b' */

      int  rowsToAdd = 2 * constraintMatrixC.getNumRows();
      int rowsBeforeAdding = constraintMatrixA.getNumRows();
      constraintMatrixA.reshape(rowsBeforeAdding + rowsToAdd, constraintMatrixA.getNumCols(), true);
      // Add C rows to matrix A
      int currRow = rowsBeforeAdding;
      for(int row = 0; row < constraintMatrixC.getNumRows(); row++)
      {
         for(int col = 0; col < constraintMatrixC.getNumCols(); col++)
         {
            constraintMatrixA.set(currRow, col, constraintMatrixC.get(row, col));
         }
         currRow++;
      }

      // Add -C rows to matrix A
      for(int row = 0; row < constraintMatrixC.getNumRows(); row++)
      {
         for(int col = 0; col < constraintMatrixC.getNumCols(); col++)
         {
            constraintMatrixA.set(currRow, col, -1 * constraintMatrixC.get(row, col));
         }
         currRow++;
      }
      augmentedInequalityMatrix = constraintMatrixA;

      //Pack b'
      int BRowsBeforeAdding = constraintVectorB.getNumRows();
      int BRowsAfterAdding = BRowsBeforeAdding + ( 2 * constraintVectorD.getNumRows() );
      constraintVectorB.reshape(BRowsAfterAdding, 1, true);

      //Add D
      int BCurrRow = BRowsBeforeAdding;
      for(int i = 0; i < constraintVectorD.getNumRows(); i++)
      {
         constraintVectorB.set(BCurrRow, 0, constraintVectorD.get(i, 0) );
         BCurrRow++;
      }
      //Add -D
      for(int i = 0; i < constraintVectorD.getNumRows(); i++)
      {
         constraintVectorB.set(BCurrRow, 0, -1 * constraintVectorD.get(i, 0) );
         BCurrRow++;
      }
      augmentedInequalityVector = constraintVectorB;

      return solve(costVectorC, augmentedInequalityMatrix, augmentedInequalityVector, solutionToPack);
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
