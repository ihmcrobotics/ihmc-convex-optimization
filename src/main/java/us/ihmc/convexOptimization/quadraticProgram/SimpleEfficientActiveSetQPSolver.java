package us.ihmc.convexOptimization.quadraticProgram;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import gnu.trove.list.array.TIntArrayList;
import us.ihmc.log.LogTools;
import us.ihmc.matrixlib.NativeMatrix;

/**
 * Solves a Quadratic Program using a simple active set method. Does not work for problems where
 * having multiple inequality constraints in the active set make the problem infeasible. Seems to
 * work well for problems with benign inequality constraints, such as variable bounds. Algorithm is
 * very fast when it can find a solution. Uses the algorithm and naming convention found in MIT
 * Paper "An efficiently solvable quadratic program for stabilizing dynamic locomotion" by Scott
 * Kuindersma, Frank Permenter, and Russ Tedrake.
 *
 * @author JerryPratt
 */
public class SimpleEfficientActiveSetQPSolver implements ActiveSetQPSolver
{
   private static final double violationFractionToAdd = 0.8;
   private static final double violationFractionToRemove = 0.95;
   //private static final double violationFractionToAdd = 1.0;
   //private static final double violationFractionToRemove = 1.0;
   private double convergenceThreshold = 1e-10;
   //private double convergenceThresholdForLagrangeMultipliers = 0.0;
   private double convergenceThresholdForLagrangeMultipliers = 1e-10;
   private int maxNumberOfIterations = 10;

   protected double quadraticCostScalar;

   private final DMatrixRMaj activeVariables = new DMatrixRMaj(0, 0);

   private final TIntArrayList activeInequalityIndices = new TIntArrayList();
   private final TIntArrayList activeUpperBoundIndices = new TIntArrayList();
   private final TIntArrayList activeLowerBoundIndices = new TIntArrayList();

   // Some temporary matrices:
   protected final NativeMatrix nativexSolutionMatrix = new NativeMatrix(0, 0);
   protected final NativeMatrix costQuadraticMatrix = new NativeMatrix(0, 0);
   protected final NativeMatrix symmetricCostQuadraticMatrix = new NativeMatrix(0, 0);

   private final NativeMatrix linearInequalityConstraintsCheck = new NativeMatrix(0, 0);

   protected final NativeMatrix quadraticCostQVector = new NativeMatrix(0, 0);
   protected final NativeMatrix quadraticCostQMatrix = new NativeMatrix(0, 0);
   protected final NativeMatrix linearEqualityConstraintsAMatrix = new NativeMatrix(0, 0);
   protected final NativeMatrix linearEqualityConstraintsBVector = new NativeMatrix(0, 0);
   
   protected final NativeMatrix linearInequalityConstraintsCMatrixO = new NativeMatrix(0, 0);
   protected final NativeMatrix linearInequalityConstraintsDVectorO = new NativeMatrix(0, 0);
   
   protected final NativeMatrix variableLowerBounds = new NativeMatrix(0, 0);
   protected final NativeMatrix variableUpperBounds = new NativeMatrix(0, 0);




   
   /** Active inequality constraints */
   private final NativeMatrix CBar = new NativeMatrix(0, 0);
   private final NativeMatrix DBar = new NativeMatrix(0, 0);
   /** Active variable bounds constraints */
   private final NativeMatrix CHat = new NativeMatrix(0, 0);
   private final NativeMatrix DHat = new NativeMatrix(0, 0);

   private final NativeMatrix QInverse = new NativeMatrix(0, 0);
   private final NativeMatrix AQInverse = new NativeMatrix(0, 0);
   private final NativeMatrix QInverseATranspose = new NativeMatrix(0, 0);
   private final NativeMatrix CBarQInverse = new NativeMatrix(0, 0);
   private final NativeMatrix CHatQInverse = new NativeMatrix(0, 0);
   private final NativeMatrix AQInverseATranspose = new NativeMatrix(0, 0);
   private final NativeMatrix AQInverseCBarTranspose = new NativeMatrix(0, 0);
   private final NativeMatrix AQInverseCHatTranspose = new NativeMatrix(0, 0);
   private final NativeMatrix CBarQInverseATranspose = new NativeMatrix(0, 0);
   private final NativeMatrix CHatQInverseATranspose = new NativeMatrix(0, 0);
   private final NativeMatrix QInverseCBarTranspose = new NativeMatrix(0, 0);
   private final NativeMatrix QInverseCHatTranspose = new NativeMatrix(0, 0);
   private final NativeMatrix CBarQInverseCBarTranspose = new NativeMatrix(0, 0);
   private final NativeMatrix CHatQInverseCHatTranspose = new NativeMatrix(0, 0);

   private final NativeMatrix CBarQInverseCHatTranspose = new NativeMatrix(0, 0);
   private final NativeMatrix CHatQInverseCBarTranspose = new NativeMatrix(0, 0);

   private final NativeMatrix AAndC = new NativeMatrix(0, 0);
   private final NativeMatrix ATransposeMuAndCTransposeLambda = new NativeMatrix(0, 0);

   private final NativeMatrix bigMatrixForLagrangeMultiplierSolution = new NativeMatrix(0, 0);
   private final NativeMatrix bigVectorForLagrangeMultiplierSolution = new NativeMatrix(0, 0);

   private final NativeMatrix tempVector = new NativeMatrix(0, 0);
   private final NativeMatrix augmentedLagrangeMultipliers = new NativeMatrix(0, 0);

   private final TIntArrayList inequalityIndicesToAddToActiveSet = new TIntArrayList();
   private final TIntArrayList inequalityIndicesToRemoveFromActiveSet = new TIntArrayList();

   private final TIntArrayList upperBoundIndicesToAddToActiveSet = new TIntArrayList();
   private final TIntArrayList upperBoundIndicesToRemoveFromActiveSet = new TIntArrayList();

   private final TIntArrayList lowerBoundIndicesToAddToActiveSet = new TIntArrayList();
   private final TIntArrayList lowerBoundIndicesToRemoveFromActiveSet = new TIntArrayList();

   protected final NativeMatrix computedObjectiveFunctionValue = new NativeMatrix(1, 1);

   private final NativeMatrix lowerBoundViolations = new NativeMatrix(0, 0);
   private final NativeMatrix upperBoundViolations = new NativeMatrix(0, 0);

   private boolean useWarmStart = false;

   private int previousNumberOfVariables = 0;
   private int previousNumberOfEqualityConstraints = 0;
   private int previousNumberOfInequalityConstraints = 0;
   private int previousNumberOfLowerBoundConstraints = 0;
   private int previousNumberOfUpperBoundConstraints = 0;

   @Override
   public void setConvergenceThreshold(double convergenceThreshold)
   {
      this.convergenceThreshold = convergenceThreshold;
   }

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

      lowerBoundViolations.reshape(0, 0);
      upperBoundViolations.reshape(0, 0);
   }

   @Override
   public void setLowerBounds(DMatrixRMaj variableLowerBounds)
   {
      if (variableLowerBounds.getNumRows() != quadraticCostQMatrix.getNumRows())
         throw new RuntimeException("variableLowerBounds.getNumRows() != quadraticCostQMatrix.getNumRows()");

      this.variableLowerBounds.set(variableLowerBounds);
   }

   @Override
   public void setUpperBounds(DMatrixRMaj variableUpperBounds)
   {
      if (variableUpperBounds.getNumRows() != quadraticCostQMatrix.getNumRows())
         throw new RuntimeException("variableUpperBounds.getNumRows() != quadraticCostQMatrix.getNumRows()");

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
      
      this.costQuadraticMatrix.set(costQuadraticMatrix);
      symmetricCostQuadraticMatrix.transpose(this.costQuadraticMatrix);

      symmetricCostQuadraticMatrix.add(this.costQuadraticMatrix, symmetricCostQuadraticMatrix);
      
      quadraticCostQMatrix.set(symmetricCostQuadraticMatrix);
      quadraticCostQMatrix.scale(0.5);


      quadraticCostQVector.set(costLinearVector);
      this.quadraticCostScalar = quadraticCostScalar;
   }

   @Override
   public double getObjectiveCost(DMatrixRMaj x)
   {
      nativexSolutionMatrix.set(x);
      
      computedObjectiveFunctionValue.multQuad(nativexSolutionMatrix, quadraticCostQMatrix);
      computedObjectiveFunctionValue.scale(0.5);
      
      computedObjectiveFunctionValue.multAddTransA(quadraticCostQVector, nativexSolutionMatrix);
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

   private final NativeMatrix lagrangeEqualityConstraintMultipliers = new NativeMatrix(0, 0);
   private final NativeMatrix lagrangeInequalityConstraintMultipliers = new NativeMatrix(0, 0);
   private final NativeMatrix lagrangeLowerBoundMultipliers = new NativeMatrix(0, 0);
   private final NativeMatrix lagrangeUpperBoundMultipliers = new NativeMatrix(0, 0);

   @Override
   public int solve(DMatrixRMaj solutionToPack)
   {
      if (!useWarmStart || problemSizeChanged())
         resetActiveSet();
      else
         addActiveSetConstraintsAsEqualityConstraints();

      int numberOfIterations = 0;

      int numberOfVariables = quadraticCostQMatrix.getNumRows();
      int numberOfEqualityConstraints = linearEqualityConstraintsAMatrix.getNumRows();
      int numberOfInequalityConstraints = linearInequalityConstraintsCMatrixO.getNumRows();
      int numberOfLowerBoundConstraints = variableLowerBounds.getNumRows();
      int numberOfUpperBoundConstraints = variableUpperBounds.getNumRows();

      nativexSolutionMatrix.reshape(numberOfVariables, 1);
      lagrangeEqualityConstraintMultipliers.reshape(numberOfEqualityConstraints, 1);
      lagrangeEqualityConstraintMultipliers.zero();
      lagrangeInequalityConstraintMultipliers.reshape(numberOfInequalityConstraints, 1);
      lagrangeInequalityConstraintMultipliers.zero();
      lagrangeLowerBoundMultipliers.reshape(numberOfLowerBoundConstraints, 1);
      lagrangeLowerBoundMultipliers.zero();
      lagrangeUpperBoundMultipliers.reshape(numberOfUpperBoundConstraints, 1);
      lagrangeUpperBoundMultipliers.zero();

      computeQInverseAndAQInverse();

      solveEqualityConstrainedSubproblemEfficiently(nativexSolutionMatrix,
                                                    lagrangeEqualityConstraintMultipliers,
                                                    lagrangeInequalityConstraintMultipliers,
                                                    lagrangeLowerBoundMultipliers,
                                                    lagrangeUpperBoundMultipliers);

      //      System.out.println(numberOfInequalityConstraints + ", " + numberOfLowerBoundConstraints + ", " + numberOfUpperBoundConstraints);
      if (numberOfInequalityConstraints == 0 && numberOfLowerBoundConstraints == 0 && numberOfUpperBoundConstraints == 0)
      {
         nativexSolutionMatrix.get(solutionToPack);
         return numberOfIterations;
      }

      for (int i = 0; i < maxNumberOfIterations; i++)
      {
         boolean activeSetWasModified = modifyActiveSetAndTryAgain(nativexSolutionMatrix,
                                                                   lagrangeEqualityConstraintMultipliers,
                                                                   lagrangeInequalityConstraintMultipliers,
                                                                   lagrangeLowerBoundMultipliers,
                                                                   lagrangeUpperBoundMultipliers);
         numberOfIterations++;

         if (!activeSetWasModified)
         {
            nativexSolutionMatrix.get(solutionToPack);
            return numberOfIterations;
         }
      }

      // No solution found. Pack NaN in all variables
      solutionToPack.reshape(numberOfVariables, 1);
      for (int i = 0; i < numberOfVariables; i++)
         solutionToPack.set(i, 0, Double.NaN);

      

      
      return numberOfIterations;
   }

   private boolean problemSizeChanged()
   {
      boolean sizeChanged = checkProblemSize();

      previousNumberOfVariables = (int) CommonOps_DDRM.elementSum(activeVariables);
      previousNumberOfEqualityConstraints = linearEqualityConstraintsAMatrix.getNumRows();
      previousNumberOfInequalityConstraints = linearInequalityConstraintsCMatrixO.getNumRows();
      previousNumberOfLowerBoundConstraints = variableLowerBounds.getNumRows();
      previousNumberOfUpperBoundConstraints = variableUpperBounds.getNumRows();

      return sizeChanged;
   }

   private boolean checkProblemSize()
   {
      if (previousNumberOfVariables != CommonOps_DDRM.elementSum(activeVariables))
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

      QInverse.invert(quadraticCostQMatrix);


      if (numberOfEqualityConstraints > 0)
      {
         AQInverse.mult(linearEqualityConstraintsAMatrix, QInverse);
         QInverseATranspose.multTransB(QInverse, linearEqualityConstraintsAMatrix);
         AQInverseATranspose.multTransB(AQInverse, linearEqualityConstraintsAMatrix);
      }
      else
      {
         AQInverse.reshape(numberOfEqualityConstraints, numberOfVariables);
         QInverseATranspose.reshape(numberOfVariables, numberOfEqualityConstraints);
         AQInverseATranspose.reshape(numberOfEqualityConstraints, numberOfEqualityConstraints);         
      }
   }

   private void computeCBarTempMatrices()
   {
      if (CBar.getNumRows() > 0)
      {
         
         AQInverseCBarTranspose.multTransB(AQInverse, CBar);

         
         CBarQInverseATranspose.mult(CBar, QInverseATranspose);

         
         CBarQInverse.mult(CBar, QInverse);

         
         QInverseCBarTranspose.multTransB(QInverse, CBar);

         
         CBarQInverseCBarTranspose.mult(CBar, QInverseCBarTranspose);
      }
      else
      {
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
         AQInverseCHatTranspose.multTransB(AQInverse, CHat);

         CHatQInverseATranspose.mult(CHat, QInverseATranspose);

         CHatQInverse.mult(CHat, QInverse);

         QInverseCHatTranspose.multTransB(QInverse, CHat);

         CHatQInverseCHatTranspose.mult(CHat, QInverseCHatTranspose);


         if (CBar.getNumRows() > 0)
         {
            CBarQInverseCHatTranspose.mult(CBar, QInverseCHatTranspose);
            CHatQInverseCBarTranspose.mult(CHat, QInverseCBarTranspose);
         }
         else
         {
            CBarQInverseCHatTranspose.reshape(CBar.getNumRows(), CHat.getNumRows());
            CHatQInverseCBarTranspose.reshape(CHat.getNumRows(), CBar.getNumRows());            
         }
      }
      else
      {
         AQInverseCHatTranspose.reshape(0, 0);
         CHatQInverseATranspose.reshape(0, 0);
         CHatQInverse.reshape(0, 0);
         QInverseCHatTranspose.reshape(0, 0);
         CHatQInverseCHatTranspose.reshape(0, 0);
         CBarQInverseCHatTranspose.reshape(0, 0);
         CHatQInverseCBarTranspose.reshape(0, 0);
      }
   }

   private boolean modifyActiveSetAndTryAgain(NativeMatrix solutionToPack, NativeMatrix lagrangeEqualityConstraintMultipliersToPack,
                                              NativeMatrix lagrangeInequalityConstraintMultipliersToPack,
                                              NativeMatrix lagrangeLowerBoundConstraintMultipliersToPack,
                                              NativeMatrix lagrangeUpperBoundConstraintMultipliersToPack)
   {
      if (solutionToPack.containsNaN())
         return false;

      boolean activeSetWasModified = false;

      // find the constraints to add
      int numberOfInequalityConstraints = linearInequalityConstraintsCMatrixO.getNumRows();
      int numberOfLowerBoundConstraints = variableLowerBounds.getNumRows();
      int numberOfUpperBoundConstraints = variableUpperBounds.getNumRows();

      double maxInequalityViolation = Double.NEGATIVE_INFINITY, maxLowerBoundViolation = Double.NEGATIVE_INFINITY,
            maxUpperBoundViolation = Double.NEGATIVE_INFINITY;
      if (numberOfInequalityConstraints != 0)
      {
         linearInequalityConstraintsCheck.scale(-1.0, linearInequalityConstraintsDVectorO);
         linearInequalityConstraintsCheck.multAdd(linearInequalityConstraintsCMatrixO, solutionToPack);

         for (int i = 0; i < linearInequalityConstraintsCheck.getNumRows(); i++)
         {
            if (activeInequalityIndices.contains(i))
               continue;

            if (linearInequalityConstraintsCheck.get(i, 0) >= maxInequalityViolation)
               maxInequalityViolation = linearInequalityConstraintsCheck.get(i, 0);
         }
      }

      if (numberOfLowerBoundConstraints != 0)
      {
         lowerBoundViolations.subtract(variableLowerBounds, solutionToPack);

         for (int i = 0; i < lowerBoundViolations.getNumRows(); i++)
         {
            if (activeLowerBoundIndices.contains(i))
               continue;

            if (lowerBoundViolations.get(i, 0) >= maxLowerBoundViolation)
               maxLowerBoundViolation = lowerBoundViolations.get(i, 0);
         }
      }
      else
      {
         lowerBoundViolations.reshape(numberOfLowerBoundConstraints, 1);         
      }

      
      if (numberOfUpperBoundConstraints != 0)
      {
         upperBoundViolations.subtract(solutionToPack, variableUpperBounds);

         for (int i = 0; i < upperBoundViolations.getNumRows(); i++)
         {
            if (activeUpperBoundIndices.contains(i))
               continue;

            if (upperBoundViolations.get(i, 0) >= maxUpperBoundViolation)
               maxUpperBoundViolation = upperBoundViolations.get(i, 0);
         }
      }
      else
      {
         upperBoundViolations.reshape(numberOfUpperBoundConstraints, 1);
      }

      double maxConstraintViolation = Math.max(maxInequalityViolation, Math.max(maxLowerBoundViolation, maxUpperBoundViolation));
      double minViolationToAdd = (1.0 - violationFractionToAdd) * maxConstraintViolation + convergenceThreshold;

      // check inequality constraints
      inequalityIndicesToAddToActiveSet.reset();
      if (maxInequalityViolation > minViolationToAdd)
      {
         for (int i = 0; i < numberOfInequalityConstraints; i++)
         {
            if (activeInequalityIndices.contains(i))
               continue; // Only check violation on those that are not active. Otherwise check should just return 0.0, but roundoff could cause problems.

            if (linearInequalityConstraintsCheck.get(i, 0) > minViolationToAdd)
            {
               activeSetWasModified = true;
               inequalityIndicesToAddToActiveSet.add(i);
            }
         }
      }

      // Check the lower bounds
      lowerBoundIndicesToAddToActiveSet.reset();
      if (maxLowerBoundViolation > minViolationToAdd)
      {
         for (int i = 0; i < numberOfLowerBoundConstraints; i++)
         {
            if (activeLowerBoundIndices.contains(i))
               continue; // Only check violation on those that are not active. Otherwise check should just return 0.0, but roundoff could cause problems.

            if (lowerBoundViolations.get(i, 0) > minViolationToAdd)
            {
               activeSetWasModified = true;
               lowerBoundIndicesToAddToActiveSet.add(i);
            }
         }
      }

      // Check the upper bounds
      upperBoundIndicesToAddToActiveSet.reset();
      if (maxUpperBoundViolation > minViolationToAdd)
      {
         for (int i = 0; i < numberOfUpperBoundConstraints; i++)
         {
            if (activeUpperBoundIndices.contains(i))
               continue; // Only check violation on those that are not active. Otherwise check should just return 0.0, but roundoff could cause problems.

            if (upperBoundViolations.get(i, 0) > minViolationToAdd)
            {
               activeSetWasModified = true;
               upperBoundIndicesToAddToActiveSet.add(i);
            }
         }
      }

      // find the constraints to remove
      int numberOfActiveInequalityConstraints = activeInequalityIndices.size();
      int numberOfActiveUpperBounds = activeUpperBoundIndices.size();
      int numberOfActiveLowerBounds = activeLowerBoundIndices.size();

      double minLagrangeInequalityMultiplier = Double.POSITIVE_INFINITY, minLagrangeLowerBoundMultiplier = Double.POSITIVE_INFINITY,
            minLagrangeUpperBoundMultiplier = Double.POSITIVE_INFINITY;

      if (numberOfActiveInequalityConstraints != 0)
         minLagrangeInequalityMultiplier = lagrangeInequalityConstraintMultipliersToPack.min();
      if (numberOfActiveLowerBounds != 0)
         minLagrangeLowerBoundMultiplier = lagrangeLowerBoundConstraintMultipliersToPack.min();
      if (numberOfActiveUpperBounds != 0)
         minLagrangeUpperBoundMultiplier = lagrangeUpperBoundConstraintMultipliersToPack.min();

      double minLagrangeMultiplier = Math.min(minLagrangeInequalityMultiplier, Math.min(minLagrangeLowerBoundMultiplier, minLagrangeUpperBoundMultiplier));
      double maxLagrangeMultiplierToRemove = -(1.0 - violationFractionToRemove) * minLagrangeMultiplier - convergenceThresholdForLagrangeMultipliers;

      inequalityIndicesToRemoveFromActiveSet.reset();
      if (minLagrangeInequalityMultiplier < maxLagrangeMultiplierToRemove)
      {
         for (int i = 0; i < activeInequalityIndices.size(); i++)
         {
            int indexToCheck = activeInequalityIndices.get(i);

            double lagrangeMultiplier = lagrangeInequalityConstraintMultipliersToPack.get(indexToCheck, 0);
            if (lagrangeMultiplier < maxLagrangeMultiplierToRemove)
            {
               activeSetWasModified = true;
               inequalityIndicesToRemoveFromActiveSet.add(indexToCheck);
            }
         }
      }

      lowerBoundIndicesToRemoveFromActiveSet.reset();
      if (minLagrangeLowerBoundMultiplier < maxLagrangeMultiplierToRemove)
      {
         for (int i = 0; i < activeLowerBoundIndices.size(); i++)
         {
            int indexToCheck = activeLowerBoundIndices.get(i);

            double lagrangeMultiplier = lagrangeLowerBoundConstraintMultipliersToPack.get(indexToCheck, 0);
            if (lagrangeMultiplier < maxLagrangeMultiplierToRemove)
            {
               activeSetWasModified = true;
               lowerBoundIndicesToRemoveFromActiveSet.add(indexToCheck);
            }
         }
      }

      upperBoundIndicesToRemoveFromActiveSet.reset();
      if (minLagrangeUpperBoundMultiplier < maxLagrangeMultiplierToRemove)
      {
         for (int i = 0; i < activeUpperBoundIndices.size(); i++)
         {
            int indexToCheck = activeUpperBoundIndices.get(i);

            double lagrangeMultiplier = lagrangeUpperBoundConstraintMultipliersToPack.get(indexToCheck, 0);
            if (lagrangeMultiplier < maxLagrangeMultiplierToRemove)
            {
               activeSetWasModified = true;
               upperBoundIndicesToRemoveFromActiveSet.add(indexToCheck);
            }
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
      addActiveSetConstraintsAsEqualityConstraints();

      solveEqualityConstrainedSubproblemEfficiently(solutionToPack,
                                                    lagrangeEqualityConstraintMultipliersToPack,
                                                    lagrangeInequalityConstraintMultipliersToPack,
                                                    lagrangeLowerBoundConstraintMultipliersToPack,
                                                    lagrangeUpperBoundConstraintMultipliersToPack);

      return true;
   }

   private void addActiveSetConstraintsAsEqualityConstraints()
   {
      int numberOfVariables = quadraticCostQMatrix.getNumRows();

      int sizeOfActiveSet = activeInequalityIndices.size();

      CBar.reshape(sizeOfActiveSet, numberOfVariables);
      DBar.reshape(sizeOfActiveSet, 1);

      for (int i = 0; i < sizeOfActiveSet; i++)
      {
         int inequalityConstraintIndex = activeInequalityIndices.get(i);
         CBar.insert(linearInequalityConstraintsCMatrixO, inequalityConstraintIndex, inequalityConstraintIndex + 1, 0, numberOfVariables, i, 0);
         DBar.insert(linearInequalityConstraintsDVectorO, inequalityConstraintIndex, inequalityConstraintIndex + 1, 0, 1, i, 0);
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
         DHat.set(row, 0, -variableLowerBounds.get(lowerBoundsConstraintIndex, 0));
         row++;
      }

      for (int i = 0; i < sizeOfUpperBoundsActiveSet; i++)
      {
         int upperBoundsConstraintIndex = activeUpperBoundIndices.get(i);

         CHat.set(row, upperBoundsConstraintIndex, 1.0);
         DHat.set(row, 0, variableUpperBounds.get(upperBoundsConstraintIndex, 0));
         row++;
      }

      //printSetChanges();
   }

   private void printSetChanges()
   {
      if (!lowerBoundIndicesToAddToActiveSet.isEmpty())
      {
         LogTools.info("Lower bound indices added : ");
         for (int i = 0; i < lowerBoundIndicesToAddToActiveSet.size(); i++)
            LogTools.info("" + lowerBoundIndicesToAddToActiveSet.get(i));
      }
      if (!lowerBoundIndicesToRemoveFromActiveSet.isEmpty())
      {
         LogTools.info("Lower bound indices removed : ");
         for (int i = 0; i < lowerBoundIndicesToRemoveFromActiveSet.size(); i++)
            LogTools.info("" + lowerBoundIndicesToRemoveFromActiveSet.get(i));
      }

      if (!upperBoundIndicesToAddToActiveSet.isEmpty())
      {
         LogTools.info("Upper bound indices added : ");
         for (int i = 0; i < upperBoundIndicesToAddToActiveSet.size(); i++)
            LogTools.info("" + upperBoundIndicesToAddToActiveSet.get(i));
      }
      if (!upperBoundIndicesToRemoveFromActiveSet.isEmpty())
      {
         LogTools.info("Upper bound indices removed : ");
         for (int i = 0; i < upperBoundIndicesToRemoveFromActiveSet.size(); i++)
            LogTools.info("" + upperBoundIndicesToRemoveFromActiveSet.get(i));
      }

      if (!inequalityIndicesToAddToActiveSet.isEmpty())
      {
         LogTools.info("Inequality constraint indices added : ");
         for (int i = 0; i < inequalityIndicesToAddToActiveSet.size(); i++)
            LogTools.info("" + inequalityIndicesToAddToActiveSet.get(i));
      }
      if (!inequalityIndicesToRemoveFromActiveSet.isEmpty())
      {
         LogTools.info("Inequality constraint indices removed : ");
         for (int i = 0; i < inequalityIndicesToRemoveFromActiveSet.size(); i++)
            LogTools.info("" + inequalityIndicesToRemoveFromActiveSet.get(i));
      }
   }

   private void solveEqualityConstrainedSubproblemEfficiently(NativeMatrix xSolutionToPack, NativeMatrix lagrangeEqualityConstraintMultipliersToPack,
                                                              NativeMatrix lagrangeInequalityConstraintMultipliersToPack,
                                                              NativeMatrix lagrangeLowerBoundConstraintMultipliersToPack,
                                                              NativeMatrix lagrangeUpperBoundConstraintMultipliersToPack)
   {
      int numberOfVariables = quadraticCostQMatrix.getNumRows();
      int numberOfOriginalEqualityConstraints = linearEqualityConstraintsAMatrix.getNumRows();

      int numberOfActiveInequalityConstraints = activeInequalityIndices.size();
      int numberOfActiveLowerBoundConstraints = activeLowerBoundIndices.size();
      int numberOfActiveUpperBoundConstraints = activeUpperBoundIndices.size();

      int numberOfAugmentedEqualityConstraints = numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints + numberOfActiveLowerBoundConstraints
            + numberOfActiveUpperBoundConstraints;

      if (numberOfAugmentedEqualityConstraints == 0)
      {
         xSolutionToPack.mult(-1.0, QInverse, quadraticCostQVector);
         return;
      }

      computeCBarTempMatrices();
      computeCHatTempMatrices();

      bigMatrixForLagrangeMultiplierSolution.reshape(numberOfAugmentedEqualityConstraints, numberOfAugmentedEqualityConstraints);
      bigVectorForLagrangeMultiplierSolution.reshape(numberOfAugmentedEqualityConstraints, 1);

      bigMatrixForLagrangeMultiplierSolution.insert(AQInverseATranspose, 0, 0);
      bigMatrixForLagrangeMultiplierSolution.insert(AQInverseCBarTranspose, 0, numberOfOriginalEqualityConstraints);
      bigMatrixForLagrangeMultiplierSolution.insert(AQInverseCHatTranspose,
                       0,
                       numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints);

      bigMatrixForLagrangeMultiplierSolution.insert(CBarQInverseATranspose, numberOfOriginalEqualityConstraints, 0);
      bigMatrixForLagrangeMultiplierSolution.insert(CBarQInverseCBarTranspose,
                       numberOfOriginalEqualityConstraints,
                       numberOfOriginalEqualityConstraints);
      bigMatrixForLagrangeMultiplierSolution.insert(CBarQInverseCHatTranspose,
                       numberOfOriginalEqualityConstraints,
                       numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints);
      bigMatrixForLagrangeMultiplierSolution.insert(CHatQInverseATranspose,
                       numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints,
                       0);
      bigMatrixForLagrangeMultiplierSolution.insert(CHatQInverseCBarTranspose,
                       numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints,
                       numberOfOriginalEqualityConstraints);
      bigMatrixForLagrangeMultiplierSolution.insert(CHatQInverseCHatTranspose,
                       numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints,
                       numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints);

      if (numberOfOriginalEqualityConstraints > 0)
      {
         bigVectorForLagrangeMultiplierSolution.insert(linearEqualityConstraintsBVector, 0, 0);
         bigVectorForLagrangeMultiplierSolution.multAddBlock(AQInverse, quadraticCostQVector, 0, 0);
      }

      if (numberOfActiveInequalityConstraints > 0)
      {
         bigVectorForLagrangeMultiplierSolution.insert(DBar, numberOfOriginalEqualityConstraints, 0);
         bigVectorForLagrangeMultiplierSolution.multAddBlock(CBarQInverse, quadraticCostQVector, numberOfOriginalEqualityConstraints, 0);
      }

      if (numberOfActiveLowerBoundConstraints + numberOfActiveUpperBoundConstraints > 0)
      {
         bigVectorForLagrangeMultiplierSolution.insert(DHat, numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints, 0);
         bigVectorForLagrangeMultiplierSolution.multAddBlock(CHatQInverse,
                                  quadraticCostQVector,
                                  numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints,
                                  0);
      }

      bigVectorForLagrangeMultiplierSolution.scale(-1.0, bigVectorForLagrangeMultiplierSolution);

      augmentedLagrangeMultipliers.solveCheck(bigMatrixForLagrangeMultiplierSolution, bigVectorForLagrangeMultiplierSolution);

      AAndC.reshape(numberOfAugmentedEqualityConstraints, numberOfVariables);
      AAndC.insert(linearEqualityConstraintsAMatrix, 0, 0);
      AAndC.insert(CBar, numberOfOriginalEqualityConstraints, 0);
      AAndC.insert(CHat, numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints, 0);

      ATransposeMuAndCTransposeLambda.multTransA(AAndC, augmentedLagrangeMultipliers);

      tempVector.add(quadraticCostQVector, ATransposeMuAndCTransposeLambda);

      xSolutionToPack.mult(-1.0, QInverse, tempVector);

      int startRow = 0;
      int numberOfRows = numberOfOriginalEqualityConstraints;
      lagrangeEqualityConstraintMultipliersToPack.insert(augmentedLagrangeMultipliers, startRow, startRow + numberOfRows, 0, 1, 0, 0);

      startRow += numberOfRows;
      lagrangeInequalityConstraintMultipliersToPack.zero();
      for (int i = 0; i < numberOfActiveInequalityConstraints; i++)
      {
         int inequalityConstraintIndex = activeInequalityIndices.get(i);
         lagrangeInequalityConstraintMultipliersToPack.insert(augmentedLagrangeMultipliers,
                           startRow + i,
                           startRow + i + 1,
                           0,
                           1,
                           inequalityConstraintIndex,
                           0);
      }

      startRow += numberOfActiveInequalityConstraints;
      lagrangeLowerBoundConstraintMultipliersToPack.zero();
      for (int i = 0; i < numberOfActiveLowerBoundConstraints; i++)
      {
         int lowerBoundConstraintIndex = activeLowerBoundIndices.get(i);
         lagrangeLowerBoundConstraintMultipliersToPack.insert(augmentedLagrangeMultipliers,
                           startRow + i,
                           startRow + i + 1,
                           0,
                           1,
                           lowerBoundConstraintIndex,
                           0);
      }

      startRow += numberOfActiveLowerBoundConstraints;
      lagrangeUpperBoundConstraintMultipliersToPack.zero();
      for (int i = 0; i < numberOfActiveUpperBoundConstraints; i++)
      {
         int upperBoundConstraintIndex = activeUpperBoundIndices.get(i);
         
         lagrangeUpperBoundConstraintMultipliersToPack.insert(augmentedLagrangeMultipliers,
                           startRow + i,
                           startRow + i + 1,
                           0,
                           1,
                           upperBoundConstraintIndex,
                           0);
      }
   }

   @Override
   public void getLagrangeEqualityConstraintMultipliers(DMatrixRMaj multipliersMatrixToPack)
   {
      lagrangeEqualityConstraintMultipliers.get(multipliersMatrixToPack);
   }

   @Override
   public void getLagrangeInequalityConstraintMultipliers(DMatrixRMaj multipliersMatrixToPack)
   {
      lagrangeInequalityConstraintMultipliers.get(multipliersMatrixToPack);
   }

   @Override
   public void getLagrangeLowerBoundsMultipliers(DMatrixRMaj multipliersMatrixToPack)
   {
      lagrangeLowerBoundMultipliers.get(multipliersMatrixToPack);
   }

   @Override
   public void getLagrangeUpperBoundsMultipliers(DMatrixRMaj multipliersMatrixToPack)
   {
      lagrangeUpperBoundMultipliers.get(multipliersMatrixToPack);
   }
}
