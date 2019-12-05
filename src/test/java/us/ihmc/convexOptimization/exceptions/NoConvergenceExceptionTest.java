package us.ihmc.convexOptimization.exceptions;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

public class NoConvergenceExceptionTest
{
   @Test
   public void testCreateAndThrowSomeNoConvergenceExceptions()
   {
      int iter = 0;

      try
      {
         throw new NoConvergenceException();
      }
      catch (NoConvergenceException e)
      {
         iter = e.getIter();
      }

      assertTrue(iter == -1, "Iter not equal -1");

      try
      {
         throw new NoConvergenceException(5);
      }
      catch (NoConvergenceException e)
      {
         iter = e.getIter();
      }

      assertTrue(iter == 5, "Iter not equal -1");
   }
}
