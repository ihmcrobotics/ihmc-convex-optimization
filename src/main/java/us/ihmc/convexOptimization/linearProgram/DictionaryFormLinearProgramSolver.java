package us.ihmc.convexOptimization.linearProgram;

import gnu.trove.list.array.TIntArrayList;
import org.apache.commons.math3.util.Precision;
import org.ejml.data.DMatrixRMaj;
import us.ihmc.commons.time.Stopwatch;

import java.util.Arrays;

/**
 * Solves a dictionary form LP using the criss-cross or simplex methods.
 * Simplex implementation borrows from org.apache.commons.math3.optim.linear.SimplexSolver and doi.org/10.3929/ethz-b-000426221 (ch. 4)
 */
public class DictionaryFormLinearProgramSolver
{
   static final boolean debug = false;

   static final int maxVariables = 200;
   static final int maxCrissCrossIterations = 50000;
   static final int maxSimplexIterations = 1000;
   static final int nullMatrixIndex = -1;
   static final double epsilon = 1e-6;
   static final double zeroCutoff = 1e-10;

   private final LinearProgramDictionary dictionary = new LinearProgramDictionary();
   private final DMatrixRMaj solution = new DMatrixRMaj(maxVariables);

   private final Stopwatch timer = new Stopwatch();
   private final SolverStatistics phase1Statistics = new SolverStatistics();
   private final SolverStatistics phase2Statistics = new SolverStatistics();
   private final SolverStatistics crissCrossStatistics = new SolverStatistics();

   private enum SimplexPhase
   {
      PHASE_I, PHASE_II;

      int objectiveSize()
      {
         return this == PHASE_I ? 2 : 1;
      }
   }

   /////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////// SIMPLEX METHOD /////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////

   public void solveSimplex(DMatrixRMaj startingDictionary)
   {
      if (startingDictionary.getNumCols() > maxVariables)
      {
         throw new IllegalArgumentException("Simplex method has a maximum of " + maxVariables + " decision variables, " + startingDictionary.getNumCols() + " provided.");
      }

      phase1Statistics.clear();
      phase2Statistics.clear();

      if (dictionary.initialize(startingDictionary, SolverMethod.SIMPLEX))
      {
         /* Phase I: compute feasible dictionary */
         performSimplexPhase(SimplexPhase.PHASE_I);

         if (!phase1Statistics.foundSolution() || dictionary.getEntry(0, 0) < -epsilon)
         {
            phase1Statistics.setFoundSolution(false);
            return;
         }

         dictionary.dropPhaseIVariables();
      }

      /* Phase II: optimize feasible dictionary */
      performSimplexPhase(SimplexPhase.PHASE_II);

      packSolution();
   }

   /* package private for testing */
   void performSimplexPhase(SimplexPhase phase)
   {
      SolverStatistics statistics = phase == SimplexPhase.PHASE_I ? phase1Statistics : phase2Statistics;

      timer.reset();

      while (true)
      {
         if (statistics.getAndIncrementIterations() > maxSimplexIterations)
         {
            statistics.setFoundSolution(false);
            break;
         }

         if (isSimplexOptimal())
         {
            statistics.setFoundSolution(true);
            break;
         }

         int s = computeSimplexPivotColumn();
         int r = computeSimplexPivotRow(s, phase);

         if (r == nullMatrixIndex)
         {
            statistics.setFoundSolution(false);
            break;
         }

         if (debug)
         {
            System.out.println("Pivoting on (" + r + "," + s + ")\n");
         }

         dictionary.performPivot(r, s);
      }

      statistics.setSolveTime(timer.lapElapsed());
   }

   private void packSolution()
   {
      solution.reshape(dictionary.getNumberOfColumns() - 1, 1);
      Arrays.fill(solution.getData(), 0.0);

      for (int i = 1; i < dictionary.getBasisSize(); i++)
      {
         int variableIndex = dictionary.getBasisIndex(i) - 1;
         if (variableIndex < solution.getNumRows())
         {
            solution.set(variableIndex, dictionary.getEntry(i, 0));
         }
      }
   }

   /* Checks optimality assuming feasibility, so only the objective row needs to be checked */
   private boolean isSimplexOptimal()
   {
      for (int j = 1; j < dictionary.getNumberOfColumns(); j++)
      {
         if (dictionary.getEntry(0, j) > epsilon)
         {
            return false;
         }
      }

      return true;
   }

   // use Bland pivot rule, which finds the positive objective row entry with the lowest corresponding variable (lexical) index
   private int computeSimplexPivotColumn()
   {
      int minimumEntryIndex = Integer.MAX_VALUE;
      int column = nullMatrixIndex;

      for (int j = 1; j < dictionary.getNumberOfColumns(); j++)
      {
         double entry = dictionary.getEntry(0, j);
         if (entry < epsilon)
         {
            continue;
         }

         int index = dictionary.getNonBasisIndex(j);
         if (index < minimumEntryIndex)
         {
            minimumEntryIndex = index;
            column = j;
         }
      }

      return column;
   }

   private final TIntArrayList minRatioIndices = new TIntArrayList(maxVariables + 1);

   private int computeSimplexPivotRow(int column, SimplexPhase phase)
   {
      double minRatio = Double.MAX_VALUE;
      minRatioIndices.reset();

      for (int i = phase.objectiveSize(); i < dictionary.getNumberOfRows(); i++)
      {
         double d_ig = dictionary.getEntry(i, 0);
         double d_is = dictionary.getEntry(i, column);

         if (d_is > -epsilon)
         {
            continue;
         }

         double ratio = Math.abs(d_ig / d_is);
         int cmp = Precision.compareTo(ratio, minRatio, epsilon);

         if (cmp == 0)
         {
            minRatioIndices.add(i);
         }
         else if (cmp < 0)
         {
            minRatioIndices.reset();
            minRatioIndices.add(i);
            minRatio = ratio;
         }
      }

      if (minRatioIndices.isEmpty())
      {
         return nullMatrixIndex;
      }
      else if (minRatioIndices.size() > 1)
      {
         // (from apache impl...)
         // apply Bland's rule to prevent cycling:
         //    take the row for which the corresponding basic variable has the smallest index

         int minRowIndex = Integer.MAX_VALUE;
         int minRow = nullMatrixIndex;

         for (int i = 0; i < minRatioIndices.size(); i++)
         {
            int variableIndex = dictionary.getBasisIndex(minRatioIndices.get(i));
            if (variableIndex < minRowIndex)
            {
               minRowIndex = variableIndex;
               minRow = minRatioIndices.get(i);
            }
         }

         return minRow;
      }
      else
      {
         return minRatioIndices.get(0);
      }
   }

   public DMatrixRMaj getSolution()
   {
      return solution;
   }

   public void printSolution()
   {
      System.out.println("Solution:");
      for (int i = 0; i < solution.getNumRows(); i++)
      {
         System.out.println("\t " + solution.get(i));
      }
   }

   /////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////// CRISS CROSS METHOD /////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////

   public void solveCrissCross(DMatrixRMaj startingDictionary)
   {
      crissCrossStatistics.clear();
      timer.reset();

      dictionary.initialize(startingDictionary, SolverMethod.CRISS_CROSS);

      while (true)
      {
         if (crissCrossStatistics.getAndIncrementIterations() > maxCrissCrossIterations)
         {
            crissCrossStatistics.setFoundSolution(false);
            break;
         }

         int candidateBasisPivot = findNegativeColumnEntryWithBlandRule(0);
         int candidateNonBasisPivot = findPositiveRowEntryWithBlandRule(0);

         int basisPivot, nonBasisPivot;
         if (candidateBasisPivot == nullMatrixIndex && candidateNonBasisPivot == nullMatrixIndex)
         {
            crissCrossStatistics.setFoundSolution(true);
            break;
         }
         else if (candidateBasisPivot != nullMatrixIndex && (candidateNonBasisPivot == nullMatrixIndex
                                                             || dictionary.getBasisIndex(candidateBasisPivot) < dictionary.getNonBasisIndex(candidateNonBasisPivot)))
         {
            basisPivot = candidateBasisPivot;
            nonBasisPivot = findPositiveRowEntryWithBlandRule(basisPivot);

            if (nonBasisPivot == nullMatrixIndex)
            {
               // inconsistent
               break;
            }
         }
         else
         {
            nonBasisPivot = candidateNonBasisPivot;
            basisPivot = findNegativeColumnEntryWithBlandRule(nonBasisPivot);

            if (basisPivot == nullMatrixIndex)
            {
               // dual inconsistent
               break;
            }
         }

         dictionary.performPivot(basisPivot, nonBasisPivot);
      }

      crissCrossStatistics.setSolveTime(timer.totalElapsed());
      packSolution();
   }

   private int findNegativeColumnEntryWithBlandRule(int column)
   {
      int minLexicalIndex = Integer.MAX_VALUE;
      int row = nullMatrixIndex;

      for (int i = 1; i < dictionary.getNumberOfRows(); i++)
      {
         double d_ig = dictionary.getEntry(i, column);
         int lexicalIndex = dictionary.getBasisIndex(i);

         if (d_ig < -epsilon && lexicalIndex < minLexicalIndex)
         {
            minLexicalIndex = lexicalIndex;
            row = i;
         }
      }

      return row;
   }

   private int findPositiveRowEntryWithBlandRule(int row)
   {
      int minLexicalIndex = Integer.MAX_VALUE;
      int column = nullMatrixIndex;

      for (int j = 1; j < dictionary.getNumberOfColumns(); j++)
      {
         double d_fj = dictionary.getEntry(row, j);
         int lexicalIndex = dictionary.getNonBasisIndex(j);

         if (d_fj > epsilon && lexicalIndex < minLexicalIndex)
         {
            minLexicalIndex = lexicalIndex;
            column = j;
         }
      }

      return column;
   }

   public SolverStatistics getPhase1Statistics()
   {
      return phase1Statistics;
   }

   public SolverStatistics getPhase2Statistics()
   {
      return phase2Statistics;
   }

   public SolverStatistics getCrissCrossStatistics()
   {
      return crissCrossStatistics;
   }
}

