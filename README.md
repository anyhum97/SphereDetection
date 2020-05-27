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
 <img src="picture2.png">
</p>

<p>
 C++ solution:<br><br>
 (-134.116, -0.664237, -5.40004): 22.7613<br>
 (15.7307, 3.03475, -7.52638): 37.9498<br>
 (-86.1651, 67.7646, -13.7158): 38.0584<br>
 (118.568, -62.2806, -1.50465): 38.3407<br>
 (-79.3969, -71.0951, -3.86699): 45.1946<br>
 (112.815, 75.961, -20.0345): 39.8314<br>
</p>
