package us.ihmc.convexOptimization.linearProgram;

import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import us.ihmc.euclid.tools.EuclidCoreTestTools;
import us.ihmc.euclid.tools.EuclidCoreTools;

public class DictionaryFormLinearProgramSolverTest
{
   // should be larger than DictionaryFormLinearProgramSolver.epsilon
   private static final double epsilon = 1e-5;

   @Test
   public void testDictionary0()
   {
      /* The example problem given in doi.org/10.3929/ethz-b-000426221 */
      DMatrixRMaj dictionary = new DMatrixRMaj(new double[] {0.0, 3.0, 4.0, 2.0, 4.0, -2.0, 0.0, 0.0, 8.0, -1.0, 0.0, -2.0, 6.0, 0.0, -3.0, -1.0});
      dictionary.reshape(4, 4);
      TDoubleArrayList expectedSolution = new TDoubleArrayList(new double[] {2.0, 1.0, 3.0});
      runTest(dictionary, expectedSolution);

      DictionaryFormLinearProgramSolver solver = new DictionaryFormLinearProgramSolver();
      solver.solveSimplex(dictionary);
      DMatrixRMaj dualSolution = solver.getDualSolution();
      Assertions.assertTrue(EuclidCoreTools.epsilonEquals(dualSolution.get(0), 4.0 / 3.0, 1e-10), "Invalid dual solution");
      Assertions.assertTrue(EuclidCoreTools.epsilonEquals(dualSolution.get(1), 1.0 / 3.0, 1e-10), "Invalid dual solution");
      Assertions.assertTrue(EuclidCoreTools.epsilonEquals(dualSolution.get(2), 4.0 / 3.0, 1e-10), "Invalid dual solution");
   }

   @Test
   public void testDictionary1()
   {
      DMatrixRMaj dictionary = new DMatrixRMaj(new double[] {0.0, 1.0, -2.0, 1.0, 0.0, -2.0, 1.0, -1.0, 0.0, -3.0, -1.0, -1.0, 0.0, 5.0, -3.0, 2.0});
      dictionary.reshape(4, 4);
      TDoubleArrayList expectedSolution = new TDoubleArrayList(new double[] {0.0, 0.0, 0.0});
      runTest(dictionary, expectedSolution);
   }

   @Test
   public void testDictionary2()
   {
      DMatrixRMaj dictionary = new DMatrixRMaj(new double[] {0.0, 1.0, 2.01, 4.0, -1.0, 0.0, 2.0, 0.0, -1.0, 6.0, -1.0, -2.0});
      dictionary.reshape(4, 3);
      TDoubleArrayList expectedSolution = new TDoubleArrayList(new double[] {2.0, 2.0});
      runTest(dictionary, expectedSolution);
   }

   @Test
   public void testDictionary3()
   {
      DMatrixRMaj dictionary = new DMatrixRMaj(new double[] {0.0, 1.0, 1.99, 4.0, -1.0, 0.0, 2.0, 0.0, -1.0, 6.0, -1.0, -2.0});
      dictionary.reshape(4, 3);
      TDoubleArrayList expectedSolution = new TDoubleArrayList(new double[] {4.0, 1.0});
      runTest(dictionary, expectedSolution);
   }

   @Test
   public void testDictionary4()
   {
      DMatrixRMaj dictionary = new DMatrixRMaj(new double[] {0.0, 1.0, -0.01, 4.0, -1.0, 0.0, 2.0, 0.0, -1.0, 6.0, -1.0, -2.0});
      dictionary.reshape(4, 3);
      TDoubleArrayList expectedSolution = new TDoubleArrayList(new double[] {4.0, 0.0});
      runTest(dictionary, expectedSolution);
   }

   @Test
   public void testDictionary5()
   {
      DMatrixRMaj dictionary = new DMatrixRMaj(new double[] {0.0, -0.01, 1.0, 4.0, -1.0, 0.0, 2.0, 0.0, -1.0, 6.0, -1.0, -2.0});
      dictionary.reshape(4, 3);
      TDoubleArrayList expectedSolution = new TDoubleArrayList(new double[] {0.0, 2.0});
      runTest(dictionary, expectedSolution);
   }

   @Test
   public void testDictionary6()
   {
      DMatrixRMaj dictionary = new DMatrixRMaj(new double[] {0.0, 2.1, 1.0, 2.0, -1.0, -1.0, -1.0, 1.0, 1.0});
      dictionary.reshape(3, 3);
      TDoubleArrayList expectedSolution = new TDoubleArrayList(new double[] {2.0, 0.0});
      runTest(dictionary, expectedSolution);
   }

   @Test
   public void testFixedBasisToyProblem()
   {
      DMatrixRMaj Ain = new DMatrixRMaj(3, 2);
      Ain.set(0, 0, 1.0);
      Ain.set(1, 1, 1.0);
      Ain.set(2, 0, 1.0);
      Ain.set(2, 1, 1.0);

      DMatrixRMaj b = new DMatrixRMaj(3, 1);
      b.set(0, 0, 2.0);
      b.set(1, 0, 2.0);
      b.set(2, 0, 3.0);

      DMatrixRMaj c = new DMatrixRMaj(2, 1);
      c.set(0, 0, 2.0);
      c.set(1, 0, 1.0);

      DMatrixRMaj solution = new DMatrixRMaj(0);

      LinearProgramSolver solver = new LinearProgramSolver();
      solver.solve(c, Ain, b, solution);

      System.out.println(solution);

      TIntArrayList basisIndices = new TIntArrayList(solver.getBasisIndices());

      // modify 3rd constraint to (1.0 + alpha1)x1 + (1.0 + alpha2)x2 <= 3
      double alpha1 = 0.1;
      double alpha2 = -0.1;
      DMatrixRMaj AinModified = new DMatrixRMaj(Ain);
      AinModified.set(2, 0, 1.0 + alpha1);
      AinModified.set(2, 1, 1.0 + alpha2);

      DMatrixRMaj solutionModified = new DMatrixRMaj(0);
      solver.solveForFixedBasis(AinModified, b, basisIndices, solutionModified);
//      System.out.println(solutionModified);
   }

   /**
    * Similar to {@link #testFixedBasisToyProblem} but goes through the solve process directly
    */
   private static void solveFixedBasisToyProblemDirectly()
   {
      // x1 <= 2, x2 <= 2, x1 + x2 <= 3, query along (2,1)

      // Ax <= b
      // A = [1,0 ; 0,1 ; 1,1]
      // b = [2 ; 2 ; 3]
      // c = [2 , 1]

      // D = [0 c^T; b -A]

      DMatrixRMaj dictionary0 = new DMatrixRMaj(new double[]{0.0, 2.0, 1.0, 2.0, -1.0, 0.0, 2.0, 0.0, -1.0, 3.0, -1.0, -1.0});
      dictionary0.reshape(4, 3);
      TDoubleArrayList solution0 = new TDoubleArrayList(new double[] {2.0, 1.0});
      runTest(dictionary0, solution0);

      // modify 3rd constraint to (1.0 + alpha1)x1 + (1.0 + alpha2)x2 <= 3
      double alpha1 = 0.1;
      double alpha2 = -0.1;

      // compute sensitivity and solution1 as a function of alpha1, alpha2...
      DMatrixRMaj A_basis = new DMatrixRMaj(new double[]{1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 + alpha1, 1.0 + alpha2, 0.0});
      A_basis.reshape(3, 3);
      DMatrixRMaj A_basis_inv = new DMatrixRMaj(0);
      CommonOps_DDRM.invert(A_basis, A_basis_inv);

      DMatrixRMaj b = new DMatrixRMaj(3, 1);
      b.setData(new double[]{2.0, 2.0, 3.0});

      DMatrixRMaj x_basis = new DMatrixRMaj(3, 1);
      CommonOps_DDRM.mult(A_basis_inv, b, x_basis);
      System.out.println("x_basis:");
      System.out.println(x_basis);

      DMatrixRMaj dictionary1 = new DMatrixRMaj(dictionary0);
      dictionary1.set(3, 1, -1.0 - alpha1);
      dictionary1.set(3, 2, -1.0 - alpha2);

      DictionaryFormLinearProgramSolver solver = new DictionaryFormLinearProgramSolver();
      solver.solveSimplex(dictionary1);
      System.out.println("with solver:");
      System.out.println(solver.getPrimalSolution());
   }

   private static void runTest(DMatrixRMaj dictionary, TDoubleArrayList expectedSolution)
   {
      DictionaryFormLinearProgramSolver solver = new DictionaryFormLinearProgramSolver();

      solver.solveCrissCross(dictionary);
      Assertions.assertTrue(solver.getCrissCrossStatistics().foundSolution());
      for (int i = 0; i < expectedSolution.size(); i++)
      {
         boolean equal = EuclidCoreTools.epsilonEquals(solver.getPrimalSolution().get(i), expectedSolution.get(i), epsilon);
         Assertions.assertTrue(equal, "Criss-cross has invalid solution");
      }

      solver.solveSimplex(dictionary);
      Assertions.assertTrue(solver.getSimplexStatistics().foundSolution());
      for (int i = 0; i < expectedSolution.size(); i++)
      {
         boolean equal = EuclidCoreTools.epsilonEquals(solver.getPrimalSolution().get(i), expectedSolution.get(i), epsilon);
         Assertions.assertTrue(equal, "Simplex has invalid solution");
      }
   }

   public static void main(String[] args)
   {
      solveFixedBasisToyProblemDirectly();
   }
}
