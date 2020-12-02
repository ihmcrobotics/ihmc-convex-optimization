plugins {
   id("us.ihmc.ihmc-build")
   id("us.ihmc.ihmc-ci") version "7.4"
   id("us.ihmc.ihmc-cd") version "1.17"
}

ihmc {
   group = "us.ihmc"
   version = "0.17.1"
   vcsUrl = "https://github.com/ihmcrobotics/ihmc-convex-optimization"
   openSource = true

   configureDependencyResolution()
   configurePublications()
}

mainDependencies {
   api("org.ejml:ejml-core:0.39")
   api("org.ejml:ejml-ddense:0.39")
   api("net.sf.trove4j:trove4j:3.0.3")

   api("org.ojalgo:ojalgo:40.0.0")
   api("com.github.vincentfk:joptimizer:3.3.0")

   api("us.ihmc:ihmc-commons:0.30.4")
   api("us.ihmc:euclid:0.15.1")
   api("us.ihmc:euclid-frame:0.15.1")
   api("us.ihmc:ihmc-matrix-library:0.18.0")
   api("us.ihmc:ihmc-native-library-loader:1.3.1")
   api("us.ihmc:ihmc-optimizer-wrappers:0.0.31")
}

testDependencies {
   api("us.ihmc:ihmc-matrix-library-test:0.18.0")
}