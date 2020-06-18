package us.ihmc.convexOptimization.quadraticProgram;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.factory.LinearSolverFactory_DDRM;
import org.ejml.interfaces.linsol.LinearSolverDense;

import gnu.trove.list.array.TIntArrayList;
import us.ihmc.matrixlib.DiagonalMatrixTools;
import us.ihmc.matrixlib.NativeCommonOps;

/**
 * Utilizes the same procedure as the Simple Efficient Active Set QP Solver, but assumes that the
 * quadratic cost matrix is diagonal.
 */
public class SimpleDiagonalActiveSetQPSolver extends SimpleEfficientActiveSetQPSolver
{
   private static final double epsilon = 1e-10;

   private int maxNumberOfIterations = 10;

   private final DMatrixRMaj quadraticCostQMatrix = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj quadraticCostQVector = new DMatrixRMaj(0, 0);
   private double quadraticCostScalar;

   private final DMatrixRMaj linearEqualityConstraintsAMatrix = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj linearEqualityConstraintsBVector = new DMatrixRMaj(0, 0);

   private final DMatrixRMaj linearInequalityConstraintsCMatrixO = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj linearInequalityConstraintsDVectorO = new DMatrixRMaj(0, 0);

   private final DMatrixRMaj variableLowerBounds = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj variableUpperBounds = new DMatrixRMaj(0, 0);

   private final TIntArrayList activeInequalityIndices = new TIntArrayList();
   private final TIntArrayList activeUpperBoundIndices = new TIntArrayList();
   private final TIntArrayList activeLowerBoundIndices = new TIntArrayList();

   // Some temporary matrices:
   private final DMatrixRMaj negativeQuadraticCostQVector = new DMatrixRMaj(0, 0);

   private final DMatrixRMaj linearInequalityConstraintsCheck = new DMatrixRMaj(0, 0);

   private final DMatrixRMaj CBar = new DMatrixRMaj(0, 0); // Active inequality constraints
   private final DMatrixRMaj DBar = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj CHat = new DMatrixRMaj(0, 0); // Active variable bounds constraints
   private final DMatrixRMaj DHat = new DMatrixRMaj(0, 0);

   private final DMatrixRMaj ATranspose = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj CBarTranspose = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj CHatTranspose = new DMatrixRMaj(0, 0);

   private final DMatrixRMaj QInverse = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj AQInverse = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj QInverseATranspose = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj CBarQInverse = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj CHatQInverse = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj AQInverseATranspose = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj AQInverseCBarTranspose = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj AQInverseCHatTranspose = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj CBarQInverseATranspose = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj CHatQInverseATranspose = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj QInverseCBarTranspose = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj QInverseCHatTranspose = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj CBarQInverseCBarTranspose = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj CHatQInverseCHatTranspose = new DMatrixRMaj(0, 0);

   private final DMatrixRMaj CBarQInverseCHatTranspose = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj CHatQInverseCBarTranspose = new DMatrixRMaj(0, 0);

   private final DMatrixRMaj ATransposeAndCTranspose = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj ATransposeMuAndCTransposeLambda = new DMatrixRMaj(0, 0);

   private final DMatrixRMaj bigMatrixForLagrangeMultiplierSolution = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj bigVectorForLagrangeMultiplierSolution = new DMatrixRMaj(0, 0);

   private final DMatrixRMaj tempVector = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj augmentedLagrangeMultipliers = new DMatrixRMaj(0, 0);

   private final TIntArrayList inequalityIndicesToAddToActiveSet = new TIntArrayList();
   private final TIntArrayList inequalityIndicesToRemoveFromActiveSet = new TIntArrayList();

   private final TIntArrayList upperBoundIndicesToAddToActiveSet = new TIntArrayList();
   private final TIntArrayList upperBoundIndicesToRemoveFromActiveSet = new TIntArrayList();

   private final TIntArrayList lowerBoundIndicesToAddToActiveSet = new TIntArrayList();
   private final TIntArrayList lowerBoundIndicesToRemoveFromActiveSet = new TIntArrayList();

   private final DMatrixRMaj computedObjectiveFunctionValue = new DMatrixRMaj(1, 1);

   private final LinearSolverDense<DMatrixRMaj> solver = LinearSolverFactory_DDRM.linear(0);

   private boolean useWarmStart = false;

   private int previousNumberOfVariables = 0;
   private int previousNumberOfEqualityConstraints = 0;
   private int previousNumberOfInequalityConstraints = 0;
   private int previousNumberOfLowerBoundConstraints = 0;
   private int previousNumberOfUpperBoundConstraints = 0;

   @Override
   public void setMaxNumberOfIterations(int maxNumberOfIterations)
   {
      this.maxNumberOfIterations = maxNumberOfIterations;
   }

   @Override
   public void clear()
   {
      quadraticCostQMatrix.reshape(0, 0);
      quadraticCostQVector.reshape(0, 0);

      linearEqualityConstraintsAMatrix.reshape(0, 0);
      linearEqualityConstraintsBVector.reshape(0, 0);

      linearInequalityConstraintsCMatrixO.reshape(0, 0);
      linearInequalityConstraintsDVectorO.reshape(0, 0);

      variableLowerBounds.reshape(0, 0);
      variableUpperBounds.reshape(0, 0);
   }

   @Override
   public void setVariableBounds(DMatrixRMaj variableLowerBounds, DMatrixRMaj variableUpperBounds)
   {
      if (variableLowerBounds.getNumRows() != quadraticCostQMatrix.getNumRows())
         throw new RuntimeException("variableLowerBounds.getNumRows() != quadraticCostQMatrix.getNumRows()");
      if (variableUpperBounds.getNumRows() != quadraticCostQMatrix.getNumRows())
         throw new RuntimeException("variableUpperBounds.getNumRows() != quadraticCostQMatrix.getNumRows()");

      this.variableLowerBounds.set(variableLowerBounds);
      this.variableUpperBounds.set(variableUpperBounds);
   }

   @Override
   public void setQuadraticCostFunction(DMatrixRMaj costQuadraticMatrix, DMatrixRMaj costLinearVector, double quadraticCostScalar)
   {
      if (costLinearVector.getNumCols() != 1)
         throw new RuntimeException("costLinearVector.getNumCols() != 1");
      if (costQuadraticMatrix.getNumRows() != costLinearVector.getNumRows())
         throw new RuntimeException("costQuadraticMatrix.getNumRows() != costLinearVector.getNumRows()");
      if (costQuadraticMatrix.getNumRows() != costQuadraticMatrix.getNumCols())
         throw new RuntimeException("costQuadraticMatrix.getNumRows() != costQuadraticMatrix.getNumCols()");

      quadraticCostQMatrix.set(costQuadraticMatrix);
      quadraticCostQVector.set(costLinearVector);
      this.quadraticCostScalar = quadraticCostScalar;
   }

   @Override
   public double getObjectiveCost(DMatrixRMaj x)
   {
      NativeCommonOps.multQuad(x, quadraticCostQMatrix, computedObjectiveFunctionValue);
      CommonOps_DDRM.scale(0.5, computedObjectiveFunctionValue);
      CommonOps_DDRM.multAddTransA(quadraticCostQVector, x, computedObjectiveFunctionValue);
      return computedObjectiveFunctionValue.get(0, 0) + quadraticCostScalar;
   }

   @Override
   public void setLinearEqualityConstraints(DMatrixRMaj linearEqualityConstraintsAMatrix, DMatrixRMaj linearEqualityConstraintsBVector)
   {
      if (linearEqualityConstraintsBVector.getNumCols() != 1)
         throw new RuntimeException("linearEqualityConstraintsBVector.getNumCols() != 1");
      if (linearEqualityConstraintsAMatrix.getNumRows() != linearEqualityConstraintsBVector.getNumRows())
         throw new RuntimeException("linearEqualityConstraintsAMatrix.getNumRows() != linearEqualityConstraintsBVector.getNumRows()");
      if (linearEqualityConstraintsAMatrix.getNumCols() != quadraticCostQMatrix.getNumCols())
         throw new RuntimeException("linearEqualityConstraintsAMatrix.getNumCols() != quadraticCostQMatrix.getNumCols()");

      this.linearEqualityConstraintsBVector.set(linearEqualityConstraintsBVector);
      this.linearEqualityConstraintsAMatrix.set(linearEqualityConstraintsAMatrix);
   }

   @Override
   public void setLinearInequalityConstraints(DMatrixRMaj linearInequalityConstraintCMatrix, DMatrixRMaj linearInequalityConstraintDVector)
   {
      if (linearInequalityConstraintDVector.getNumCols() != 1)
         throw new RuntimeException("linearInequalityConstraintDVector.getNumCols() != 1");
      if (linearInequalityConstraintCMatrix.getNumRows() != linearInequalityConstraintDVector.getNumRows())
         throw new RuntimeException("linearInequalityConstraintCMatrix.getNumRows() != linearInequalityConstraintDVector.getNumRows()");
      if (linearInequalityConstraintCMatrix.getNumCols() != quadraticCostQMatrix.getNumCols())
         throw new RuntimeException("linearInequalityConstraintCMatrix.getNumCols() != quadraticCostQMatrix.getNumCols()");

      linearInequalityConstraintsDVectorO.set(linearInequalityConstraintDVector);
      linearInequalityConstraintsCMatrixO.set(linearInequalityConstraintCMatrix);
   }

   @Override
   public void setUseWarmStart(boolean useWarmStart)
   {
      this.useWarmStart = useWarmStart;
   }

   @Override
   public void resetActiveSet()
   {
      CBar.reshape(0, 0);
      CHat.reshape(0, 0);
      DBar.reshape(0, 0);
      DHat.reshape(0, 0);

      activeInequalityIndices.reset();
      activeUpperBoundIndices.reset();
      activeLowerBoundIndices.reset();
   }

   private final DMatrixRMaj lagrangeEqualityConstraintMultipliers = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj lagrangeInequalityConstraintMultipliers = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj lagrangeLowerBoundMultipliers = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj lagrangeUpperBoundMultipliers = new DMatrixRMaj(0, 0);

   @Override
   public int solve(DMatrixRMaj solutionToPack)
   {
      if (!useWarmStart || problemSizeChanged())
         resetActiveSet();

      int numberOfIterations = 0;

      int numberOfVariables = quadraticCostQMatrix.getNumRows();
      int numberOfEqualityConstraints = linearEqualityConstraintsAMatrix.getNumRows();
      int numberOfInequalityConstraints = linearInequalityConstraintsCMatrixO.getNumRows();
      int numberOfLowerBoundConstraints = variableLowerBounds.getNumRows();
      int numberOfUpperBoundConstraints = variableUpperBounds.getNumRows();

      solutionToPack.reshape(numberOfVariables, 1);
      lagrangeEqualityConstraintMultipliers.reshape(numberOfEqualityConstraints, 1);
      lagrangeEqualityConstraintMultipliers.zero();
      lagrangeInequalityConstraintMultipliers.reshape(numberOfInequalityConstraints, 1);
      lagrangeInequalityConstraintMultipliers.zero();
      lagrangeLowerBoundMultipliers.reshape(numberOfLowerBoundConstraints, 1);
      lagrangeLowerBoundMultipliers.zero();
      lagrangeUpperBoundMultipliers.reshape(numberOfUpperBoundConstraints, 1);
      lagrangeUpperBoundMultipliers.zero();

      computeQInverseAndAQInverse();

      solveEqualityConstrainedSubproblemEfficiently(solutionToPack,
                                                    lagrangeEqualityConstraintMultipliers,
                                                    lagrangeInequalityConstraintMultipliers,
                                                    lagrangeLowerBoundMultipliers,
                                                    lagrangeUpperBoundMultipliers);

      //      System.out.println(numberOfInequalityConstraints + ", " + numberOfLowerBoundConstraints + ", " + numberOfUpperBoundConstraints);
      if (numberOfInequalityConstraints == 0 && numberOfLowerBoundConstraints == 0 && numberOfUpperBoundConstraints == 0)
         return numberOfIterations;

      // Test the inequality constraints:

      for (int i = 0; i < maxNumberOfIterations; i++)
      {
         boolean activeSetWasModified = modifyActiveSetAndTryAgain(solutionToPack,
                                                                   lagrangeEqualityConstraintMultipliers,
                                                                   lagrangeInequalityConstraintMultipliers,
                                                                   lagrangeLowerBoundMultipliers,
                                                                   lagrangeUpperBoundMultipliers);
         numberOfIterations++;

         if (!activeSetWasModified)
            return numberOfIterations;
      }

      for (int i = 0; i < numberOfVariables; i++)
         solutionToPack.set(i, 0, Double.NaN);

      return numberOfIterations;
   }

   private boolean problemSizeChanged()
   {
      boolean sizeChanged = checkProblemSize();

      previousNumberOfVariables = quadraticCostQMatrix.getNumRows();
      previousNumberOfEqualityConstraints = linearEqualityConstraintsAMatrix.getNumRows();
      previousNumberOfInequalityConstraints = linearInequalityConstraintsCMatrixO.getNumRows();
      previousNumberOfLowerBoundConstraints = variableLowerBounds.getNumRows();
      previousNumberOfUpperBoundConstraints = variableUpperBounds.getNumRows();

      return sizeChanged;
   }

   private boolean checkProblemSize()
   {
      if (previousNumberOfVariables != quadraticCostQMatrix.getNumRows())
         return true;
      if (previousNumberOfEqualityConstraints != linearEqualityConstraintsAMatrix.getNumRows())
         return true;
      if (previousNumberOfInequalityConstraints != linearInequalityConstraintsCMatrixO.getNumRows())
         return true;
      if (previousNumberOfLowerBoundConstraints != variableLowerBounds.getNumRows())
         return true;
      if (previousNumberOfUpperBoundConstraints != variableUpperBounds.getNumRows())
         return true;

      return false;
   }

   private void computeQInverseAndAQInverse()
   {
      int numberOfVariables = quadraticCostQMatrix.getNumRows();
      int numberOfEqualityConstraints = linearEqualityConstraintsAMatrix.getNumRows();

      ATranspose.reshape(linearEqualityConstraintsAMatrix.getNumCols(), linearEqualityConstraintsAMatrix.getNumRows());
      CommonOps_DDRM.transpose(linearEqualityConstraintsAMatrix, ATranspose);
      QInverse.reshape(numberOfVariables, numberOfVariables);

      AQInverse.reshape(numberOfEqualityConstraints, numberOfVariables);
      QInverseATranspose.reshape(numberOfVariables, numberOfEqualityConstraints);
      AQInverseATranspose.reshape(numberOfEqualityConstraints, numberOfEqualityConstraints);

      DiagonalMatrixTools.invertDiagonalMatrix(quadraticCostQMatrix, QInverse);

      if (numberOfEqualityConstraints > 0)
      {
         DiagonalMatrixTools.postMult(linearEqualityConstraintsAMatrix, QInverse, AQInverse);
         DiagonalMatrixTools.preMult(QInverse, ATranspose, QInverseATranspose);
         CommonOps_DDRM.mult(AQInverse, ATranspose, AQInverseATranspose);
      }
   }

   private void computeCBarTempMatrices()
   {
      if (CBar.getNumRows() > 0)
      {
         CBarTranspose.reshape(CBar.getNumCols(), CBar.getNumRows());
         CommonOps_DDRM.transpose(CBar, CBarTranspose);

         AQInverseCBarTranspose.reshape(AQInverse.getNumRows(), CBarTranspose.getNumCols());
         CommonOps_DDRM.mult(AQInverse, CBarTranspose, AQInverseCBarTranspose);

         CBarQInverseATranspose.reshape(CBar.getNumRows(), QInverseATranspose.getNumCols());
         CommonOps_DDRM.mult(CBar, QInverseATranspose, CBarQInverseATranspose);

         CBarQInverse.reshape(CBar.getNumRows(), QInverse.getNumCols());
         DiagonalMatrixTools.postMult(CBar, QInverse, CBarQInverse);

         QInverseCBarTranspose.reshape(QInverse.getNumRows(), CBarTranspose.getNumCols());
         DiagonalMatrixTools.preMult(QInverse, CBarTranspose, QInverseCBarTranspose);

         CBarQInverseCBarTranspose.reshape(CBar.getNumRows(), QInverseCBarTranspose.getNumCols());
         CommonOps_DDRM.mult(CBar, QInverseCBarTranspose, CBarQInverseCBarTranspose);
      }
      else
      {
         CBarTranspose.reshape(0, 0);
         AQInverseCBarTranspose.reshape(0, 0);
         CBarQInverseATranspose.reshape(0, 0);
         CBarQInverse.reshape(0, 0);
         QInverseCBarTranspose.reshape(0, 0);
         CBarQInverseCBarTranspose.reshape(0, 0);
      }
   }

   private void computeCHatTempMatrices()
   {
      if (CHat.getNumRows() > 0)
      {
         CHatTranspose.reshape(CHat.getNumCols(), CHat.getNumRows());
         CommonOps_DDRM.transpose(CHat, CHatTranspose);

         AQInverseCHatTranspose.reshape(AQInverse.getNumRows(), CHatTranspose.getNumCols());
         CommonOps_DDRM.mult(AQInverse, CHatTranspose, AQInverseCHatTranspose);

         CHatQInverseATranspose.reshape(CHat.getNumRows(), QInverseATranspose.getNumCols());
         CommonOps_DDRM.mult(CHat, QInverseATranspose, CHatQInverseATranspose);

         CHatQInverse.reshape(CHat.getNumRows(), QInverse.getNumCols());
         DiagonalMatrixTools.postMult(CHat, QInverse, CHatQInverse);

         QInverseCHatTranspose.reshape(QInverse.getNumRows(), CHatTranspose.getNumCols());
         DiagonalMatrixTools.preMult(QInverse, CHatTranspose, QInverseCHatTranspose);

         CHatQInverseCHatTranspose.reshape(CHat.getNumRows(), QInverseCHatTranspose.getNumCols());
         CommonOps_DDRM.mult(CHat, QInverseCHatTranspose, CHatQInverseCHatTranspose);

         CBarQInverseCHatTranspose.reshape(CBar.getNumRows(), CHat.getNumRows());
         CHatQInverseCBarTranspose.reshape(CHat.getNumRows(), CBar.getNumRows());

         if (CBar.getNumRows() > 0)
         {
            CommonOps_DDRM.mult(CBar, QInverseCHatTranspose, CBarQInverseCHatTranspose);
            CommonOps_DDRM.mult(CHat, QInverseCBarTranspose, CHatQInverseCBarTranspose);
         }
      }
      else
      {
         CHatTranspose.reshape(0, 0);
         AQInverseCHatTranspose.reshape(0, 0);
         CHatQInverseATranspose.reshape(0, 0);
         CHatQInverse.reshape(0, 0);
         QInverseCHatTranspose.reshape(0, 0);
         CHatQInverseCHatTranspose.reshape(0, 0);
         CBarQInverseCHatTranspose.reshape(0, 0);
         CHatQInverseCBarTranspose.reshape(0, 0);
      }
   }

   private boolean modifyActiveSetAndTryAgain(DMatrixRMaj solutionToPack, DMatrixRMaj lagrangeEqualityConstraintMultipliersToPack,
                                              DMatrixRMaj lagrangeInequalityConstraintMultipliersToPack,
                                              DMatrixRMaj lagrangeLowerBoundConstraintMultipliersToPack,
                                              DMatrixRMaj lagrangeUpperBoundConstraintMultipliersToPack)
   {
      if (containsNaN(solutionToPack))
         return false;

      boolean activeSetWasModified = false;

      int numberOfVariables = quadraticCostQMatrix.getNumRows();
      //      int numberOfEqualityConstraints = linearEqualityConstraintsAMatrix.getNumRows();
      int numberOfInequalityConstraints = linearInequalityConstraintsCMatrixO.getNumRows();
      int numberOfLowerBoundConstraints = variableLowerBounds.getNumRows();
      int numberOfUpperBoundConstraints = variableUpperBounds.getNumRows();

      inequalityIndicesToAddToActiveSet.reset();
      inequalityIndicesToRemoveFromActiveSet.reset();
      if (numberOfInequalityConstraints != 0)
      {

         linearInequalityConstraintsCheck.reshape(numberOfInequalityConstraints, 1);
         CommonOps_DDRM.mult(linearInequalityConstraintsCMatrixO, solutionToPack, linearInequalityConstraintsCheck);
         CommonOps_DDRM.subtractEquals(linearInequalityConstraintsCheck, linearInequalityConstraintsDVectorO);

         for (int i = 0; i < numberOfInequalityConstraints; i++)
         {
            if (activeInequalityIndices.contains(i))
               continue; // Only check violation on those that are not active. Otherwise check should just return 0.0, but roundoff could cause problems.
            if (linearInequalityConstraintsCheck.get(i, 0) > epsilon)
            {
               activeSetWasModified = true;
               inequalityIndicesToAddToActiveSet.add(i);
            }
         }

         for (int i = 0; i < activeInequalityIndices.size(); i++)
         {
            int indexToCheck = activeInequalityIndices.get(i);

            double lagrangeMultiplier = lagrangeInequalityConstraintMultipliersToPack.get(indexToCheck);
            if (lagrangeMultiplier < 0.0)
            {
               activeSetWasModified = true;
               inequalityIndicesToRemoveFromActiveSet.add(indexToCheck);
            }
         }
      }

      // Check the Bounds
      lowerBoundIndicesToAddToActiveSet.reset();
      for (int i = 0; i < numberOfLowerBoundConstraints; i++)
      {
         if (activeLowerBoundIndices.contains(i))
            continue; // Only check violation on those that are not active. Otherwise check should just return 0.0, but roundoff could cause problems.

         double solutionVariable = solutionToPack.get(i, 0);
         double lowerBound = variableLowerBounds.get(i, 0);
         if (solutionVariable < lowerBound - epsilon)
         {
            activeSetWasModified = true;
            lowerBoundIndicesToAddToActiveSet.add(i);
         }
      }

      upperBoundIndicesToAddToActiveSet.reset();
      for (int i = 0; i < numberOfUpperBoundConstraints; i++)
      {
         if (activeUpperBoundIndices.contains(i))
            continue; // Only check violation on those that are not active. Otherwise check should just return 0.0, but roundoff could cause problems.

         double solutionVariable = solutionToPack.get(i, 0);
         double upperBound = variableUpperBounds.get(i, 0);
         if (solutionVariable > upperBound + epsilon)
         {
            activeSetWasModified = true;
            upperBoundIndicesToAddToActiveSet.add(i);
         }
      }

      lowerBoundIndicesToRemoveFromActiveSet.reset();
      for (int i = 0; i < activeLowerBoundIndices.size(); i++)
      {
         int indexToCheck = activeLowerBoundIndices.get(i);

         double lagrangeMultiplier = lagrangeLowerBoundConstraintMultipliersToPack.get(indexToCheck);
         if (lagrangeMultiplier < 0.0)
         {
            activeSetWasModified = true;
            lowerBoundIndicesToRemoveFromActiveSet.add(indexToCheck);
         }
      }

      upperBoundIndicesToRemoveFromActiveSet.reset();
      for (int i = 0; i < activeUpperBoundIndices.size(); i++)
      {
         int indexToCheck = activeUpperBoundIndices.get(i);

         double lagrangeMultiplier = lagrangeUpperBoundConstraintMultipliersToPack.get(indexToCheck);
         //         if ((lagrangeMultiplier < 0) || (Double.isInfinite(lagrangeMultiplier)))
         if (lagrangeMultiplier < 0.0)
         {
            activeSetWasModified = true;
            upperBoundIndicesToRemoveFromActiveSet.add(indexToCheck);
         }
      }

      if (!activeSetWasModified)
         return false;

      for (int i = 0; i < inequalityIndicesToAddToActiveSet.size(); i++)
      {
         activeInequalityIndices.add(inequalityIndicesToAddToActiveSet.get(i));
      }
      for (int i = 0; i < inequalityIndicesToRemoveFromActiveSet.size(); i++)
      {
         activeInequalityIndices.remove(inequalityIndicesToRemoveFromActiveSet.get(i));
      }

      for (int i = 0; i < lowerBoundIndicesToAddToActiveSet.size(); i++)
      {
         activeLowerBoundIndices.add(lowerBoundIndicesToAddToActiveSet.get(i));
      }
      for (int i = 0; i < lowerBoundIndicesToRemoveFromActiveSet.size(); i++)
      {
         activeLowerBoundIndices.remove(lowerBoundIndicesToRemoveFromActiveSet.get(i));
      }

      for (int i = 0; i < upperBoundIndicesToAddToActiveSet.size(); i++)
      {
         activeUpperBoundIndices.add(upperBoundIndicesToAddToActiveSet.get(i));
      }
      for (int i = 0; i < upperBoundIndicesToRemoveFromActiveSet.size(); i++)
      {
         activeUpperBoundIndices.remove(upperBoundIndicesToRemoveFromActiveSet.get(i));
      }

      // Add active set constraints as equality constraints:
      int sizeOfActiveSet = activeInequalityIndices.size();

      CBar.reshape(sizeOfActiveSet, numberOfVariables);
      DBar.reshape(sizeOfActiveSet, 1);

      for (int i = 0; i < sizeOfActiveSet; i++)
      {
         int inequalityConstraintIndex = activeInequalityIndices.get(i);
         CommonOps_DDRM.extract(linearInequalityConstraintsCMatrixO, inequalityConstraintIndex, inequalityConstraintIndex + 1, 0, numberOfVariables, CBar, i, 0);
         CommonOps_DDRM.extract(linearInequalityConstraintsDVectorO, inequalityConstraintIndex, inequalityConstraintIndex + 1, 0, 1, DBar, i, 0);
      }

      // Add active bounds constraints as equality constraints:
      int sizeOfLowerBoundsActiveSet = activeLowerBoundIndices.size();
      int sizeOfUpperBoundsActiveSet = activeUpperBoundIndices.size();

      int sizeOfBoundsActiveSet = sizeOfLowerBoundsActiveSet + sizeOfUpperBoundsActiveSet;

      CHat.reshape(sizeOfBoundsActiveSet, numberOfVariables);
      DHat.reshape(sizeOfBoundsActiveSet, 1);

      CHat.zero();
      DHat.zero();

      int row = 0;

      for (int i = 0; i < sizeOfLowerBoundsActiveSet; i++)
      {
         int lowerBoundsConstraintIndex = activeLowerBoundIndices.get(i);

         CHat.set(row, lowerBoundsConstraintIndex, -1.0);
         DHat.set(row, 0, -variableLowerBounds.get(lowerBoundsConstraintIndex));
         row++;
      }

      for (int i = 0; i < sizeOfUpperBoundsActiveSet; i++)
      {
         int upperBoundsConstraintIndex = activeUpperBoundIndices.get(i);

         CHat.set(row, upperBoundsConstraintIndex, 1.0);
         DHat.set(row, 0, variableUpperBounds.get(upperBoundsConstraintIndex));
         row++;
      }

      solveEqualityConstrainedSubproblemEfficiently(solutionToPack,
                                                    lagrangeEqualityConstraintMultipliersToPack,
                                                    lagrangeInequalityConstraintMultipliersToPack,
                                                    lagrangeLowerBoundConstraintMultipliersToPack,
                                                    lagrangeUpperBoundConstraintMultipliersToPack);

      return true;
   }

   private boolean containsNaN(DMatrixRMaj solution)
   {
      for (int i = 0; i < solution.getNumRows(); i++)
      {
         if (Double.isNaN(solution.get(i, 0)))
            return true;
      }

      return false;
   }

   private void solveEqualityConstrainedSubproblemEfficiently(DMatrixRMaj xSolutionToPack, DMatrixRMaj lagrangeEqualityConstraintMultipliersToPack,
                                                              DMatrixRMaj lagrangeInequalityConstraintMultipliersToPack,
                                                              DMatrixRMaj lagrangeLowerBoundConstraintMultipliersToPack,
                                                              DMatrixRMaj lagrangeUpperBoundConstraintMultipliersToPack)
   {
      int numberOfVariables = quadraticCostQMatrix.getNumRows();
      int numberOfOriginalEqualityConstraints = linearEqualityConstraintsAMatrix.getNumRows();

      int numberOfActiveInequalityConstraints = activeInequalityIndices.size();
      int numberOfActiveLowerBoundConstraints = activeLowerBoundIndices.size();
      int numberOfActiveUpperBoundConstraints = activeUpperBoundIndices.size();

      int numberOfAugmentedEqualityConstraints = numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints + numberOfActiveLowerBoundConstraints
            + numberOfActiveUpperBoundConstraints;

      negativeQuadraticCostQVector.set(quadraticCostQVector);
      CommonOps_DDRM.scale(-1.0, negativeQuadraticCostQVector);

      if (numberOfAugmentedEqualityConstraints == 0)
      {
         DiagonalMatrixTools.preMult(QInverse, negativeQuadraticCostQVector, xSolutionToPack);
         return;
      }

      computeCBarTempMatrices();
      computeCHatTempMatrices();

      bigMatrixForLagrangeMultiplierSolution.reshape(numberOfAugmentedEqualityConstraints, numberOfAugmentedEqualityConstraints);
      bigVectorForLagrangeMultiplierSolution.reshape(numberOfAugmentedEqualityConstraints, 1);

      CommonOps_DDRM.insert(AQInverseATranspose, bigMatrixForLagrangeMultiplierSolution, 0, 0);
      CommonOps_DDRM.insert(AQInverseCBarTranspose, bigMatrixForLagrangeMultiplierSolution, 0, numberOfOriginalEqualityConstraints);
      CommonOps_DDRM.insert(AQInverseCHatTranspose,
                       bigMatrixForLagrangeMultiplierSolution,
                       0,
                       numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints);

      CommonOps_DDRM.insert(CBarQInverseATranspose, bigMatrixForLagrangeMultiplierSolution, numberOfOriginalEqualityConstraints, 0);
      CommonOps_DDRM.insert(CBarQInverseCBarTranspose,
                       bigMatrixForLagrangeMultiplierSolution,
                       numberOfOriginalEqualityConstraints,
                       numberOfOriginalEqualityConstraints);
      CommonOps_DDRM.insert(CBarQInverseCHatTranspose,
                       bigMatrixForLagrangeMultiplierSolution,
                       numberOfOriginalEqualityConstraints,
                       numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints);

      CommonOps_DDRM.insert(CHatQInverseATranspose,
                       bigMatrixForLagrangeMultiplierSolution,
                       numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints,
                       0);
      CommonOps_DDRM.insert(CHatQInverseCBarTranspose,
                       bigMatrixForLagrangeMultiplierSolution,
                       numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints,
                       numberOfOriginalEqualityConstraints);
      CommonOps_DDRM.insert(CHatQInverseCHatTranspose,
                       bigMatrixForLagrangeMultiplierSolution,
                       numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints,
                       numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints);

      if (numberOfOriginalEqualityConstraints > 0)
      {
         tempVector.reshape(numberOfOriginalEqualityConstraints, 1);
         CommonOps_DDRM.mult(AQInverse, quadraticCostQVector, tempVector);
         CommonOps_DDRM.addEquals(tempVector, linearEqualityConstraintsBVector);
         CommonOps_DDRM.scale(-1.0, tempVector);

         CommonOps_DDRM.insert(tempVector, bigVectorForLagrangeMultiplierSolution, 0, 0);
      }

      if (numberOfActiveInequalityConstraints > 0)
      {
         tempVector.reshape(numberOfActiveInequalityConstraints, 1);
         CommonOps_DDRM.mult(CBarQInverse, quadraticCostQVector, tempVector);
         CommonOps_DDRM.addEquals(tempVector, DBar);
         CommonOps_DDRM.scale(-1.0, tempVector);

         CommonOps_DDRM.insert(tempVector, bigVectorForLagrangeMultiplierSolution, numberOfOriginalEqualityConstraints, 0);
      }

      if (numberOfActiveLowerBoundConstraints + numberOfActiveUpperBoundConstraints > 0)
      {
         tempVector.reshape(numberOfActiveLowerBoundConstraints + numberOfActiveUpperBoundConstraints, 1);
         CommonOps_DDRM.mult(CHatQInverse, quadraticCostQVector, tempVector);
         CommonOps_DDRM.addEquals(tempVector, DHat);
         CommonOps_DDRM.scale(-1.0, tempVector);

         CommonOps_DDRM.insert(tempVector, bigVectorForLagrangeMultiplierSolution, numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints, 0);
      }

      augmentedLagrangeMultipliers.reshape(numberOfAugmentedEqualityConstraints, 1);
      solver.setA(bigMatrixForLagrangeMultiplierSolution);
      solver.solve(bigVectorForLagrangeMultiplierSolution, augmentedLagrangeMultipliers);

      ATransposeAndCTranspose.reshape(numberOfVariables, numberOfAugmentedEqualityConstraints);
      CommonOps_DDRM.insert(ATranspose, ATransposeAndCTranspose, 0, 0);
      CommonOps_DDRM.insert(CBarTranspose, ATransposeAndCTranspose, 0, numberOfOriginalEqualityConstraints);
      CommonOps_DDRM.insert(CHatTranspose, ATransposeAndCTranspose, 0, numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints);

      ATransposeMuAndCTransposeLambda.reshape(numberOfVariables, 1);
      CommonOps_DDRM.mult(ATransposeAndCTranspose, augmentedLagrangeMultipliers, ATransposeMuAndCTransposeLambda);

      tempVector.set(quadraticCostQVector);
      CommonOps_DDRM.scale(-1.0, tempVector);
      CommonOps_DDRM.subtractEquals(tempVector, ATransposeMuAndCTransposeLambda);

      DiagonalMatrixTools.preMult(QInverse, tempVector, xSolutionToPack);

      int startRow = 0;
      int numberOfRows = numberOfOriginalEqualityConstraints;
      CommonOps_DDRM.extract(augmentedLagrangeMultipliers, startRow, startRow + numberOfRows, 0, 1, lagrangeEqualityConstraintMultipliersToPack, 0, 0);

      startRow += numberOfRows;
      lagrangeInequalityConstraintMultipliersToPack.zero();
      for (int i = 0; i < numberOfActiveInequalityConstraints; i++)
      {
         int inequalityConstraintIndex = activeInequalityIndices.get(i);
         CommonOps_DDRM.extract(augmentedLagrangeMultipliers,
                           startRow + i,
                           startRow + i + 1,
                           0,
                           1,
                           lagrangeInequalityConstraintMultipliersToPack,
                           inequalityConstraintIndex,
                           0);
      }

      startRow += numberOfActiveInequalityConstraints;
      lagrangeLowerBoundConstraintMultipliersToPack.zero();
      for (int i = 0; i < numberOfActiveLowerBoundConstraints; i++)
      {
         int lowerBoundConstraintIndex = activeLowerBoundIndices.get(i);
         CommonOps_DDRM.extract(augmentedLagrangeMultipliers,
                           startRow + i,
                           startRow + i + 1,
                           0,
                           1,
                           lagrangeLowerBoundConstraintMultipliersToPack,
                           lowerBoundConstraintIndex,
                           0);
      }

      startRow += numberOfActiveLowerBoundConstraints;
      lagrangeUpperBoundConstraintMultipliersToPack.zero();
      for (int i = 0; i < numberOfActiveUpperBoundConstraints; i++)
      {
         int upperBoundConstraintIndex = activeUpperBoundIndices.get(i);
         CommonOps_DDRM.extract(augmentedLagrangeMultipliers,
                           startRow + i,
                           startRow + i + 1,
                           0,
                           1,
                           lagrangeUpperBoundConstraintMultipliersToPack,
                           upperBoundConstraintIndex,
                           0);
      }
   }

   @Override
   public void getLagrangeEqualityConstraintMultipliers(DMatrixRMaj multipliersMatrixToPack)
   {
      multipliersMatrixToPack.set(lagrangeEqualityConstraintMultipliers);
   }

   @Override
   public void getLagrangeInequalityConstraintMultipliers(DMatrixRMaj multipliersMatrixToPack)
   {
      multipliersMatrixToPack.set(lagrangeInequalityConstraintMultipliers);
   }

   @Override
   public void getLagrangeLowerBoundsMultipliers(DMatrixRMaj multipliersMatrixToPack)
   {
      multipliersMatrixToPack.set(lagrangeLowerBoundMultipliers);
   }

   @Override
   public void getLagrangeUpperBoundsMultipliers(DMatrixRMaj multipliersMatrixToPack)
   {
      multipliersMatrixToPack.set(lagrangeUpperBoundMultipliers);
   }
}
