plugins {
   id("us.ihmc.ihmc-build")
}

ihmc {
   group = "us.ihmc"
   version = "0.18.0"
   vcsUrl = "https://github.com/ihmcrobotics/ihmc-convex-optimization"
   openSource = true

   configureDependencyResolution()
   configurePublications()
}

mainDependencies {
   api("org.ejml:ejml-core:0.39")
   api("org.ejml:ejml-ddense:0.39")
   api("net.sf.trove4j:trove4j:3.0.3")

   api("org.ojalgo:ojalgo:49.2.1")
   api("com.github.vincentfk:joptimizer:3.3.0")
   {
      exclude(group = "log4j", module = "log4j")
   }

   api("us.ihmc:ihmc-commons:0.35.1")
   api("us.ihmc:euclid:0.22.5")
   api("us.ihmc:euclid-frame:0.22.5")
   api("us.ihmc:ihmc-matrix-library:0.19.3")
   api("us.ihmc:ihmc-native-library-loader:2.0.6")
   api("us.ihmc:ihmc-optimizer-wrappers:0.1.1")
}

testDependencies {
   api("us.ihmc:ihmc-matrix-library-test:0.19.3")
}
