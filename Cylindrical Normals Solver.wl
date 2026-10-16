ClearAll[makeTransverseBasis, cylindricalInputQ, solveCylindricalShock];

makeTransverseBasis[direction_List] := Module[{seed, e1, e2},
  seed = If[Abs[direction . {0., 0., 1.}] < 0.9, {0., 0., 1.}, {0., 1., 0.}];
  e1 = Normalize[Cross[seed, direction]];
  e2 = Normalize[Cross[direction, e1]];
  {e1, e2}
];

cylindricalInputQ[data_Association] := Module[
  {positions, relativeTimes, direction, times},
  positions = Lookup[data, "positionsRE", Missing["positionsRE"]];
  relativeTimes = Lookup[data, "relativeTimesSeconds", Missing["relativeTimesSeconds"]];
  direction = Lookup[data, "propagationDirectionGSE", Missing["propagationDirectionGSE"]];
  times = Prepend[N[relativeTimes], 0.];
  Dimensions[positions] === {6, 3} && MatrixQ[positions, NumericQ] &&
    Length[relativeTimes] === 5 && VectorQ[relativeTimes, NumericQ] &&
    AllTrue[Differences[times], # > 0. &] && Length[direction] === 3 &&
    VectorQ[direction, NumericQ] && Norm[N[direction]] > 0.
];

solveCylindricalShock::params = "Expected six time-ordered GSE position vectors in R_E, five positive relative crossing times in seconds, and a nonzero GSE propagation vector.";
solveCylindricalShock::branch = "No positive, subluminal real cylindrical solution was found.";

solveCylindricalShock[data_Association] := Module[
  {positions, times, direction, referencePosition, displacements, e1, e2,
   transverseCoordinates, longitudinalCoordinates, offset, speed, alpha, beta,
   quadraticU2, quadraticUV, quadraticV2, variables, crossingEquations,
   rankOneEquation, solutions, physicalSolutions, makeBranch, branches},

  If[!cylindricalInputQ[data], Message[solveCylindricalShock::params]; Return[$Failed]];

  positions = N[data["positionsRE"]];
  times = Prepend[N[data["relativeTimesSeconds"]], 0.];
  direction = Normalize[N[data["propagationDirectionGSE"]]];
  referencePosition = First[positions];
  displacements = (# - referencePosition &) /@ positions;
  {e1, e2} = makeTransverseBasis[direction];
  transverseCoordinates = ({# . e1, # . e2} &) /@ displacements;
  longitudinalCoordinates = (# . direction &) /@ displacements;

  variables = {offset, speed, alpha, beta, quadraticU2, quadraticUV, quadraticV2};
  crossingEquations = MapThread[
    #1 == offset + speed #2 + alpha #3[[1]] + beta #3[[2]] +
       quadraticU2 #3[[1]]^2 + quadraticUV #3[[1]] #3[[2]] +
       quadraticV2 #3[[2]]^2 &,
    {longitudinalCoordinates, times, transverseCoordinates}
  ];
  rankOneEquation = quadraticUV^2 == 4. quadraticU2 quadraticV2;
  solutions = NSolve[Join[crossingEquations, {rankOneEquation}], variables, Reals];
  physicalSolutions = Select[
    solutions,
    TrueQ[0. < (speed /. #) < 299792.458/6378.] &
  ];
  If[physicalSolutions === {}, Message[solveCylindricalShock::branch]; Return[$Failed]];

  makeBranch[solution_] := Module[
    {parameterValues, offsetValue, speedValue, alphaValue, betaValue,
     quadraticU2Value, quadraticUVValue, quadraticV2Value, hessian,
     eigenvalues, eigenvectors, curvatureIndex, curvatureValue,
     curvatureAxisTransverse, curvatureAxisAngleRadians, curvatureAxisGSE, tiltVectorGSE,
     transverseAxisGSE, cylinderAxisGSE, unnormalizedNormals, normals,
     residuals, quadraticTerms, pairwiseNormalAnglesDegrees},

    parameterValues = N[variables /. solution];
    {offsetValue, speedValue, alphaValue, betaValue, quadraticU2Value,
     quadraticUVValue, quadraticV2Value} = parameterValues;
    hessian = {{2. quadraticU2Value, quadraticUVValue},
      {quadraticUVValue, 2. quadraticV2Value}};
    {eigenvalues, eigenvectors} = Eigensystem[hessian];
    curvatureIndex = First@Ordering[Abs[eigenvalues], -1];
    curvatureValue = eigenvalues[[curvatureIndex]];
    curvatureAxisTransverse = Normalize[eigenvectors[[curvatureIndex]]];
    If[curvatureAxisTransverse[[1]] < 0., curvatureAxisTransverse = -curvatureAxisTransverse];
    curvatureAxisAngleRadians = ArcTan[curvatureAxisTransverse[[1]], curvatureAxisTransverse[[2]]];
    curvatureAxisGSE = Normalize[curvatureAxisTransverse[[1]] e1 + curvatureAxisTransverse[[2]] e2];
    tiltVectorGSE = alphaValue e1 + betaValue e2;
    transverseAxisGSE = Normalize[Cross[direction, curvatureAxisGSE]];
    cylinderAxisGSE = Normalize[transverseAxisGSE + (tiltVectorGSE . transverseAxisGSE) direction];
    unnormalizedNormals = Map[
      direction - tiltVectorGSE -
        (hessian . #)[[1]] e1 - (hessian . #)[[2]] e2 &,
      transverseCoordinates
    ];
    normals = Normalize /@ unnormalizedNormals;
    residuals = MapThread[
      #1 - (offsetValue + speedValue #2 + alphaValue #3[[1]] + betaValue #3[[2]] +
         quadraticU2Value #3[[1]]^2 + quadraticUVValue #3[[1]] #3[[2]] +
         quadraticV2Value #3[[2]]^2) &,
      {longitudinalCoordinates, times, transverseCoordinates}
    ];
    quadraticTerms = (0.5 # . hessian . # &) /@ transverseCoordinates;
    pairwiseNormalAnglesDegrees = (180./Pi ArcCos[Clip[#[[1]] . #[[2]], {-1., 1.}]] &) /@
      Subsets[normals, {2}];

    <|
      "offsetRE" -> offsetValue,
      "speedREPerSecond" -> speedValue,
      "speedKmPerSecond" -> 6378. speedValue,
      "sixParameterFit" -> <|
        "offsetRE" -> offsetValue,
        "speedREPerSecond" -> speedValue,
        "tiltE1" -> alphaValue,
        "tiltE2" -> betaValue,
        "curvatureREInverse" -> curvatureValue,
        "curvatureAxisAngleRadians" -> curvatureAxisAngleRadians
      |>,
      "propagationDirectionGSE" -> direction,
      "transverseBasisGSE" -> {e1, e2},
      "tiltCoefficients" -> {alphaValue, betaValue},
      "tiltVectorGSE" -> tiltVectorGSE,
      "quadraticCoefficients" -> {quadraticU2Value, quadraticUVValue, quadraticV2Value},
      "curvatureREInverse" -> curvatureValue,
      "curvatureAxisAngleRadians" -> curvatureAxisAngleRadians,
      "curvatureAxisAngleDegrees" -> 180./Pi curvatureAxisAngleRadians,
      "curvatureAxisGSE" -> curvatureAxisGSE,
      "cylinderAxisGSE" -> cylinderAxisGSE,
      "localNormalsGSE" -> normals,
      "localNormalSpeedsKmPerSecond" -> (6378. speedValue (direction . #) & /@ normals),
      "maxPairwiseNormalAngleDegrees" -> Max[pairwiseNormalAnglesDegrees],
      "maxAbsQuadraticTermRE" -> Max[Abs[quadraticTerms]],
      "residualsRE" -> residuals,
      "maxAbsResidualRE" -> Max[Abs[residuals]]
    |>
  ];

  branches = SortBy[makeBranch /@ physicalSolutions, Abs[#["curvatureREInverse"]] &];

  <|
    "model" -> "tilted_parabolic_cylinder",
    "referencePositionRE" -> referencePosition,
    "relativeTimesSeconds" -> times,
    "transverseCoordinatesRE" -> transverseCoordinates,
    "longitudinalCoordinatesRE" -> longitudinalCoordinates,
    "branchOrder" -> "ascending_absolute_curvature",
    "geometricDefaultBranchIndex" -> 1,
    "geometricDefaultConvention" -> "minimum_absolute_curvature",
    "branches" -> branches
  |>
];

repoRoot = DirectoryName[$InputFileName];
event = "2023-02-26 19-23-00";
shockParamsPath = FileNameJoin[{repoRoot, "Shocks", "Shock Params.json"}];
shockNormalsPath = FileNameJoin[{repoRoot, "Shocks", "Kinematic Shock Normals (Wolfram).json"}];
shockParams = Import[shockParamsPath, "RawJSON"];
shockNormals = Import[shockNormalsPath, "RawJSON"];
eventParams = shockParams[event];
satellites = SortBy[
  Keys[eventParams],
  AbsoluteTime[DateObject[eventParams[#]["t0"], TimeZone -> 0]] &
];
crossingTimes = DateObject[eventParams[#]["t0"], TimeZone -> 0] & /@ satellites;
referenceTime = First[crossingTimes];
relativeTimesSeconds = Rest[AbsoluteTime[#] - AbsoluteTime[referenceTime] & /@ crossingTimes];
positionsRE = Lookup[eventParams[#]["position"], {"x", "y", "z"}] & /@ satellites;
propagationDirectionGSE = First[Values[shockNormals[event]["planar"]]]["propagation_direction_gse"];

fit = solveCylindricalShock[<|
  "positionsRE" -> positionsRE,
  "relativeTimesSeconds" -> relativeTimesSeconds,
  "propagationDirectionGSE" -> propagationDirectionGSE
|>];

output = <|
  "event" -> event,
  "satellites" -> satellites,
  "referenceSatellite" -> First[satellites],
  "referenceTime" -> DateString[referenceTime, {"ISODate", "T", "Time", "Z"}],
  "positionsRE" -> positionsRE,
  "sourceShockParams" -> shockParamsPath,
  "sourcePropagationDirection" -> shockNormalsPath,
  "fit" -> fit
|>;

outputPath = FileNameJoin[{repoRoot, "Shocks", "Cylindrical Shock 2023-02-26 (Wolfram).json"}];
Export[outputPath, output, "RawJSON"];
Print["Exported: " <> outputPath];
Print["Physical branches: " <> ToString[Length[fit["branches"]]]];
