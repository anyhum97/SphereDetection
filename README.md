# SphereDetection
<p>
 Sphere detection algorithm:<br>

 <p>
 Environment:<br><br>
 1. Programming language could be C/C++/OpenCL/CUDA.<br>
 2. Code compatible with Microsoft Visual C++ Compiler 2017.<br>
 3. Third-party libraries only Eigen (http://eigen.tuxfamily.org/), OpenMesh (http://www.openmesh.org/).<br>
 </p>

 <p>
 Task description:<br><br>
 Develop an application that receives a polygonal mesh with known topology, without single-disconnected points, maximal number of spheres, and fits the given number of spheres in the input mesh.<br>
 Additional parameters could be RMS distance threshold from the detected spheres.<br>
 The spheres can have different radius.<br>
 The application outputs the center and radius for each sphere.<br>
 </p>

 <p>
 Additional requirements:<br><br>
 - For a given example mesh algorithm should fit 5 spheres in less then 1 second (excluding I/O operations) on CPU Core i7 3Ghz)<br>
 - Production-grade code<br>
 - Cross-platform solution<br>
 </p>
</p>

<p>
 <img src="picture1.png">
</p>

<p>
 <h1>Solution<br>
  
 <img src="picture2.png"><br>
  <center><h4>colored map of z-projection on XY plane<h3></center><br>
  
Look at colored map of z-projection on XY plane. You can see that the centers of the spheres correspond to local maximum
or minimum if the spheres were concave. There we will look for points belonging to the sphere. Then we can check if these points are points of a sphere using https://mathworld.wolfram.com/Circumsphere.html<br>  
</p>

<p>
Note that we take the projection onto the XY plane because we know the topology of the problem.<br>
</p>

<p>
There are three projects in this repository:<br><br>
 <p>
 1) C++ Monte-Carlo solution (Cpp Implementation):<br><br>
 use only 5% of vertex-points. Works fast but may contain artifacts.<br><br>
 </p>
 
<p>
 2) Base C++ solution (Cpp Base):<br><br>
 use all vertex-points. Is the basis for cuda implementation. Works slowly.<br><br>
</p>
 
<p>
 3) Cuda solution (Cuda Implementation):<br><br>
 use all vertex-points. Works fast.<br><br>
</p>

</p>
