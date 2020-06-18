package us.ihmc.convexOptimization.experimental;

import java.util.ArrayList;

import org.ejml.data.DMatrixRMaj;

import com.joptimizer.functions.ConvexMultivariateRealFunction;

public interface ExperimentalSOCPSolver
{
   // Minimize f^T x
   public void setOptimizationFunctionVectorF(double[] optimizationFunctionVectorF);

   public void setOptimizationFunctionVectorF(DMatrixRMaj optimizationFunctionVectorF);

   /*
    * A x = b
    */
   public void setLinearEqualityConstraints(double[][] linearEqualityAMatrix, double[] linearEqualityBVector);

   public void setLinearEqualityConstraints(DMatrixRMaj linearEqualityAMatrix, DMatrixRMaj linearEqualityBVector);

   /*
    * || B x || <= u^T x
    */
   public void setSpecialSecondOrderConeInequality(double[][] coneInequalityMatrixB, double[] coneInequalityVectorU,
                                                   ArrayList<ConvexMultivariateRealFunction> otherInequalities);

   public void setSpecialSecondOrderConeInequality(DMatrixRMaj coneInequalityMatrixB, DMatrixRMaj coneInequalityVectorU,
                                                   ArrayList<ConvexMultivariateRealFunction> otherInequalities);

   public double[] solveAndReturnOptimalVector();
}
