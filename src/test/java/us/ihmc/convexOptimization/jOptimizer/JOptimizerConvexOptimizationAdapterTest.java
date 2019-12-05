package us.ihmc.convexOptimization.jOptimizer;

import org.junit.jupiter.api.Disabled;

import us.ihmc.convexOptimization.ConvexOptimizationAdapter;
import us.ihmc.convexOptimization.ConvexOptimizationAdapterTest;

//TODO: Get this working some day!!
@Disabled
public class JOptimizerConvexOptimizationAdapterTest extends ConvexOptimizationAdapterTest
{
   @Override
   public ConvexOptimizationAdapter createConvexOptimizationAdapter()
   {
      return new JOptimizerConvexOptimizationAdapter();
   }

   @Override
   public double getTestErrorEpsilon()
   {
      return 1e-5;
   }

}
