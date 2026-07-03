// BuildScript.cs — CLI player builds for the three desktop OSes (M0-10-T10).
//
// Invoke from the command line, e.g.:
//   Unity -batchmode -quit -projectPath examples/unity-demo \
//         -executeMethod BuildScript.BuildLinux64
//
// The matching Build Support module for the target OS must be installed in the
// invoking Editor. scripting backend defaults to Mono; switch to IL2CPP in
// Player Settings (the demo's callback design is already IL2CPP-safe — T05).

#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

public static class BuildScript
{
    private const string Scene = "Assets/Scenes/VokraDemo.unity";

    private static string[] Scenes => new[] { Scene };

    public static void BuildMacOS() => Build(BuildTarget.StandaloneOSX, "Build/macOS/VokraDemo.app");

    public static void BuildWindows64() => Build(BuildTarget.StandaloneWindows64, "Build/Windows/VokraDemo.exe");

    public static void BuildLinux64() => Build(BuildTarget.StandaloneLinux64, "Build/Linux/VokraDemo");

    private static void Build(BuildTarget target, string outPath)
    {
        string dir = Path.GetDirectoryName(outPath);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir);
        }

        var options = new BuildPlayerOptions
        {
            scenes = Scenes,
            target = target,
            locationPathName = outPath,
            options = BuildOptions.None,
        };

        BuildReport report = BuildPipeline.BuildPlayer(options);
        BuildSummary summary = report.summary;

        if (summary.result == BuildResult.Succeeded)
        {
            Debug.Log($"[vokra-demo] build ok: {outPath} ({summary.totalSize} bytes)");
            EditorApplication.Exit(0);
        }
        else
        {
            Debug.LogError($"[vokra-demo] build {summary.result}: {summary.totalErrors} error(s)");
            EditorApplication.Exit(1);
        }
    }
}
#endif
