package us.ihmc.convexOptimization.linearProgram;

import gnu.trove.list.array.TIntArrayList;
import org.ejml.data.DMatrixRMaj;
import us.ihmc.euclid.tools.EuclidCoreIOTools;
import us.ihmc.matrixlib.MatrixTools;

import java.util.Arrays;

import static us.ihmc.convexOptimization.linearProgram.DictionaryFormLinearProgramSolver.*;

/**
 * The dictionary used by {@link DictionaryFormLinearProgramSolver}. See doi.org/10.3929/ethz-b-000426221 for details.
 *
 * A linear program in dictionary form is represented as
 * <p>
 * x<sub>B</sub> = D x<sub>N</sub>
 * </p>
 * Where D is the dictionary, x<sub>B</sub> is the basis vector and x<sub>N</sub> is the non-basis. Both x vectors are non-negative.
 */
public class LinearProgramDictionary
{
   private static final int nullMatrixIndex = -1;
   private static final int rhsVariableLexicalIndex = 0;
   private static final int objectiveLexicalIndex = -1;
   private static final int auxObjectiveLexicalIndex = -2;

   private DMatrixRMaj dictionary = new DMatrixRMaj(maxVariables + 1, maxVariables + 1);
   private DMatrixRMaj tempDictionary = new DMatrixRMaj(maxVariables + 1, maxVariables + 1);
   private DMatrixRMaj startingDictionary;

   private final TIntArrayList basisIndices = new TIntArrayList(maxVariables);
   private final TIntArrayList nonBasisIndices = new TIntArrayList(maxVariables);

   private final TIntArrayList auxiliaryIndices = new TIntArrayList(maxVariables);
   private final TIntArrayList initialNegativeBasisIndices = new TIntArrayList(maxVariables);

   /**
    * If criss-cross is requested, the dictionary is initialized and this returns false.
    * <p>
    * If simplex is requested, it checks for feasibility.
    * If non-feasible a Phase I dictionary is setup and this returns true. Otherwise a Phase II dictionary is setup and this returns false.
    *
    * @return whether to perform Simplex Phase I on the given dictionary.
    */
   public boolean initialize(DMatrixRMaj startingDictionary, SolverMethod solverMethod)
   {
      this.startingDictionary = startingDictionary;

      if (solverMethod == SolverMethod.SIMPLEX)
      {
         setupIndexLists(startingDictionary);
         if (populateNegativeBasisIndices(startingDictionary) > 0)
         {
            // Starting dictionary not feasible, setup for Phase I
            int auxiliaryVariables = initialNegativeBasisIndices.size();
            dictionary.reshape(startingDictionary.getNumRows() + 1, startingDictionary.getNumCols() + auxiliaryVariables);
            tempDictionary.reshape(startingDictionary.getNumRows() + 1, startingDictionary.getNumCols() + auxiliaryVariables);

            Arrays.fill(dictionary.getData(), 0.0);
            MatrixTools.setMatrixBlock(dictionary, 1, 0, startingDictionary, 0, 0, startingDictionary.getNumRows(), startingDictionary.getNumCols(), 1.0);

            basisIndices.insert(0, auxObjectiveLexicalIndex);
            auxiliaryIndices.reset();
            int maximumNonAuxiliaryLexicalIndex = basisIndices.get(basisIndices.size() - 1);
            for (int i = 0; i < initialNegativeBasisIndices.size(); i++)
            {
               int auxiliaryLexicalIndex = maximumNonAuxiliaryLexicalIndex + i + 1;
               auxiliaryIndices.add(auxiliaryLexicalIndex);
               nonBasisIndices.add(auxiliaryLexicalIndex);
               dictionary.set(0, startingDictionary.getNumCols() + i, -1.0);
               dictionary.set(initialNegativeBasisIndices.get(i) + 1, startingDictionary.getNumCols() + i, 1.0);
            }

            if (debug)
            {
               printDictionary("Phase I initial auxiliary dictionary");
            }

            // Pivot out negative b entries to make feasible
            for (int i = 0; i < initialNegativeBasisIndices.size(); i++)
            {
               performPivot(initialNegativeBasisIndices.get(i) + 1, startingDictionary.getNumCols() + i);
            }

            if (debug)
            {
               System.out.println();
               printDictionary("Phase I feasible auxiliary dictionary");
            }

            return true;
         }
         else
         {
            // Starting dictionary feasible, setup for Phase II
            dictionary.set(startingDictionary);
            tempDictionary.set(startingDictionary);
            setupIndexLists(startingDictionary);

            return false;
         }
      }
      else
      {
         // Setup for Criss-cross
         dictionary.set(startingDictionary);
         tempDictionary.set(startingDictionary);
         setupIndexLists(startingDictionary);
         return false;
      }
   }

   public boolean dropPhaseIVariables()
   {
      // pivot out auxiliary indices if in basis
      for (int i = 0; i < auxiliaryIndices.size(); i++)
      {
         if (basisIndices.contains(auxiliaryIndices.get(i)))
         {
            int pivotRow = basisIndices.indexOf(auxiliaryIndices.get(i));
            int pivotColumn = findLargestMagnitudeNonAuxiliaryRowEntry(pivotRow);
            performPivot(pivotRow, pivotColumn);
         }
      }

      tempDictionary.reshape(startingDictionary.getNumRows(), startingDictionary.getNumCols());
      Arrays.fill(tempDictionary.getData(), 0.0);

      // remove auxiliary variables from dictionary
      int column = 0;
      for (int i = 0; i < dictionary.getNumCols(); i++)
      {
         int index = nonBasisIndices.get(i);
         if (!auxiliaryIndices.contains(index))
         {
            MatrixTools.setMatrixBlock(tempDictionary, 0, column, dictionary, 1, i, tempDictionary.getNumRows(), 1, 1.0);
            column++;
         }
      }

      dictionary.set(tempDictionary);

      // remove auxiliary variables from index list
      for (int i = nonBasisIndices.size() - 1; i >= 0; i--)
      {
         if (auxiliaryIndices.contains(nonBasisIndices.get(i)))
         {
            nonBasisIndices.removeAt(i);
         }
      }
      basisIndices.remove(auxObjectiveLexicalIndex);

      return true;
   }

   // Avoid pivoting on really small entries by just taking largest magnitude
   private int findLargestMagnitudeNonAuxiliaryRowEntry(int row)
   {
      double largestMagnitudeValue = 0.0;
      int column = nullMatrixIndex;

      for (int j = 1; j < dictionary.getNumCols(); j++)
      {
         int index = nonBasisIndices.get(j);
         if (auxiliaryIndices.contains(index))
         {
            continue;
         }

         double value = Math.abs(dictionary.get(row, j));
         if (value > largestMagnitudeValue)
         {
            largestMagnitudeValue = value;
            column = j;
         }
      }

      return column;
   }

   // r = basisPivot
   // s = nonBasisPivot
   public void performPivot(int r, int s)
   {
      /* Pivot is performed on temp dictionary */
      for (int i = 0; i < dictionary.getNumRows(); i++)
      {
         for (int j = 0; j < dictionary.getNumCols(); j++)
         {
            if (i == r && j == s)
            {
               tempDictionary.set(i, j, 1.0 / dictionary.get(r, s));
            }
            else if (i == r)
            {
               tempDictionary.set(i, j, -dictionary.get(r, j) / (dictionary.get(r, s)));
            }
            else if (j == s)
            {
               tempDictionary.set(i, j, dictionary.get(i, s) / (dictionary.get(r, s)));
            }
            else
            {
               tempDictionary.set(i, j, dictionary.get(i, j) - dictionary.get(i, s) * dictionary.get(r, j) / (dictionary.get(r, s)));
            }
         }
      }

      /* Update index mapping */
      int originalBasisIndex = basisIndices.get(r);
      int originalNonBasisIndex = nonBasisIndices.get(s);
      basisIndices.set(r, originalNonBasisIndex);
      nonBasisIndices.set(s, originalBasisIndex);

      /* Swap dictionaries to avoid calling .set() */
      DMatrixRMaj previousDictionary = dictionary;
      dictionary = tempDictionary;
      tempDictionary = previousDictionary;
   }

   private int populateNegativeBasisIndices(DMatrixRMaj dictionary)
   {
      initialNegativeBasisIndices.reset();
      for (int i = 1; i < dictionary.getNumRows(); i++)
      {
         if (dictionary.get(i, 0) < -zeroCutoff)
         {
            initialNegativeBasisIndices.add(i);
         }
      }

      return initialNegativeBasisIndices.size();
   }

   private void setupIndexLists(DMatrixRMaj dictionary)
   {
      basisIndices.reset();
      nonBasisIndices.reset();

      nonBasisIndices.add(rhsVariableLexicalIndex);
      basisIndices.add(objectiveLexicalIndex);

      int lexicalIndex = 1;

      for (int i = 1; i < dictionary.getNumCols(); i++)
      {
         nonBasisIndices.add(lexicalIndex++);
      }

      for (int i = 1; i < dictionary.getNumRows(); i++)
      {
         basisIndices.add(lexicalIndex++);
      }
   }

   public int getBasisSize()
   {
      return basisIndices.size();
   }

   public int getNonBasisSize()
   {
      return nonBasisIndices.size();
   }

   public int getNumberOfColumns()
   {
      return dictionary.getNumCols();
   }

   public int getNumberOfRows()
   {
      return dictionary.getNumRows();
   }

   public double getEntry(int row, int column)
   {
      return dictionary.get(row, column);
   }

   public int getBasisIndex(int dictionaryRow)
   {
      return basisIndices.get(dictionaryRow);
   }

   public int getNonBasisIndex(int dictionaryColumn)
   {
      return nonBasisIndices.get(dictionaryColumn);
   }

   private static final String entryFormat = EuclidCoreIOTools.getStringFormat(6, 3);
   private static final String entrySeparator = "\t\t";

   void printDictionary(String label)
   {
      System.out.println(label);

      for (int row = -1; row < dictionary.getNumRows(); row++)
      {
         for (int column = -1; column < dictionary.getNumCols(); column++)
         {
            String entry = "";
            if (row == -1 && column == -1)
            {

            }
            else if (row == -1)
            {
               entry = formatIndex(nonBasisIndices.get(column)) + "\t";
            }
            else if (column == -1)
            {
               entry = formatIndex(basisIndices.get(row));
            }
            else
            {
               entry = String.format(entryFormat, dictionary.get(row, column));
            }

            System.out.print(entry + entrySeparator);
         }
         System.out.println();
      }
   }

   private static String formatIndex(int index)
   {
      if (index == rhsVariableLexicalIndex)
      {
         return "g";
      }
      else if (index == objectiveLexicalIndex)
      {
         return "f";
      }
      else if (index == auxObjectiveLexicalIndex)
      {
         return "f'";
      }
      else
      {
         return Integer.toString(index);
      }
   }
}
