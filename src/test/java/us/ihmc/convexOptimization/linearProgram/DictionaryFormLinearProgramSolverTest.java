package us.ihmc.convexOptimization.linearProgram;

import gnu.trove.list.array.TDoubleArrayList;
import org.ejml.data.DMatrixRMaj;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import us.ihmc.euclid.tools.EuclidCoreTools;

public class DictionaryFormLinearProgramSolverTest
{
   // should be larger than DictionaryFormLinearProgramSolver.epsilon
   private static final double epsilon = 1e-5;

   @Test
   public void testDictionary0()
   {
      DMatrixRMaj dictionary = new DMatrixRMaj(new double[]{0.0, 3.0, 4.0, 2.0,
                                                            4.0, -2.0, 0.0, 0.0,
                                                            8.0, -1.0, 0.0, -2.0,
                                                            6.0, 0.0, -3.0, -1.0});
      dictionary.reshape(4, 4);
      TDoubleArrayList expectedSolution = new TDoubleArrayList(new double[]{2.0, 1.0, 3.0});
      runTest(dictionary, expectedSolution);
   }

   @Test
   public void testDictionary1()
   {
      DMatrixRMaj dictionary = new DMatrixRMaj(new double[]{0.0, 1.0, -2.0, 1.0,
                                                            0.0, -2.0, 1.0, -1.0,
                                                            0.0, -3.0, -1.0, -1.0,
                                                            0.0, 5.0, -3.0, 2.0});
      dictionary.reshape(4, 4);
      TDoubleArrayList expectedSolution = new TDoubleArrayList(new double[]{0.0, 0.0, 0.0});
      runTest(dictionary, expectedSolution);
   }

   @Test
   public void testDictionary2()
   {
      DMatrixRMaj dictionary = new DMatrixRMaj(new double[]{0.0, 1.0, 2.01,
                                                            4.0, -1.0, 0.0,
                                                            2.0, 0.0, -1.0,
                                                            6.0, -1.0, -2.0});
      dictionary.reshape(4, 3);
      TDoubleArrayList expectedSolution = new TDoubleArrayList(new double[]{2.0, 2.0});
      runTest(dictionary, expectedSolution);
   }

   @Test
   public void testDictionary3()
   {
      DMatrixRMaj dictionary = new DMatrixRMaj(new double[]{0.0, 1.0, 1.99,
                                                            4.0, -1.0, 0.0,
                                                            2.0, 0.0, -1.0,
                                                            6.0, -1.0, -2.0});
      dictionary.reshape(4, 3);
      TDoubleArrayList expectedSolution = new TDoubleArrayList(new double[]{4.0, 1.0});
      runTest(dictionary, expectedSolution);
   }

   @Test
   public void testDictionary4()
   {
      DMatrixRMaj dictionary = new DMatrixRMaj(new double[]{0.0, 1.0, -0.01,
                                                            4.0, -1.0, 0.0,
                                                            2.0, 0.0, -1.0,
                                                            6.0, -1.0, -2.0});
      dictionary.reshape(4, 3);
      TDoubleArrayList expectedSolution = new TDoubleArrayList(new double[]{4.0, 0.0});
      runTest(dictionary, expectedSolution);
   }

   @Test
   public void testDictionary5()
   {
      DMatrixRMaj dictionary = new DMatrixRMaj(new double[]{0.0, -0.01, 1.0,
                                                            4.0, -1.0, 0.0,
                                                            2.0, 0.0, -1.0,
                                                            6.0, -1.0, -2.0});
      dictionary.reshape(4, 3);
      TDoubleArrayList expectedSolution = new TDoubleArrayList(new double[]{0.0, 2.0});
      runTest(dictionary, expectedSolution);
   }

   @Test
   public void testDictionary6()
   {
      DMatrixRMaj dictionary = new DMatrixRMaj(new double[]{0.0, 2.1, 1.0,
                                                            2.0, -1.0, -1.0,
                                                            -1.0, 1.0, 1.0});
      dictionary.reshape(3, 3);
      TDoubleArrayList expectedSolution = new TDoubleArrayList(new double[]{2.0, 0.0});
      runTest(dictionary, expectedSolution);
   }

   private void runTest(DMatrixRMaj dictionary, TDoubleArrayList expectedSolution)
   {
      DictionaryFormLinearProgramSolver solver = new DictionaryFormLinearProgramSolver();

      solver.solveCrissCross(dictionary);
      Assertions.assertTrue(solver.getCrissCrossStatistics().foundSolution());
      for (int i = 0; i < expectedSolution.size(); i++)
      {
         boolean equal = EuclidCoreTools.epsilonEquals(solver.getSolution().get(i), expectedSolution.get(i), epsilon);
         Assertions.assertTrue(equal, "Criss-cross has invalid solution");
      }

      solver.solveSimplex(dictionary);
      Assertions.assertTrue(solver.getPhase2Statistics().foundSolution());
      for (int i = 0; i < expectedSolution.size(); i++)
      {
         boolean equal = EuclidCoreTools.epsilonEquals(solver.getSolution().get(i), expectedSolution.get(i), epsilon);
         Assertions.assertTrue(equal, "Simplex has invalid solution");
      }
   }
}
