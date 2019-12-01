package us.ihmc.convexOptimization.randomSearch;

import us.ihmc.convexOptimization.ConvexOptimizationAdapter;
import us.ihmc.convexOptimization.ConvexOptimizationAdapterTest;

public class RandomSearchConvexOptimizationAdapterTest extends ConvexOptimizationAdapterTest
{
   @Override
   public ConvexOptimizationAdapter createConvexOptimizationAdapter()
   {
      return new RandomSearchConvexOptimizationAdapter();
   }

   @Override
   public double getTestErrorEpsilon()
   {
      return 0.02;
   }
}
