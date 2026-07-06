// Vokra iOS PostProcessBuild hook.
//
// Wires the static libvokra.a (produced by M2-02's Vokra.xcframework, device
// slice extracted by scripts/collect-ios-lib.sh into Plugins/iOS/libvokra.a)
// into the generated Xcode project:
//
//   (a) Adds libvokra.a to the "Link Binary With Libraries" phase of the
//       UnityFramework target (defensive; the .meta importer usually handles
//       this, but we assert it here so a broken importer doesn't silently
//       ship a runtime that fails at first P/Invoke with a symbol-not-found
//       crash on device).
//   (b) Links Metal.framework and Accelerate.framework (non-weak). These are
//       the authoritative framework dependencies declared in the M2-02 iOS
//       build spec — Metal is required by the Metal backend feature (M2-01)
//       when enabled; Accelerate provides vImage/vDSP fallbacks used by the
//       CPU backend's ARM64 NEON dispatch on iOS.
//   (c) Disables bitcode (ENABLE_BITCODE=NO). Apple deprecated bitcode in
//       Xcode 14 (2022); leaving it enabled with a bitcode-less static
//       archive fails the link phase on older Xcode installs.
//   (d) Fails the build if OTHER_LDFLAGS contains "-undefined dynamic_lookup".
//       That flag defers symbol resolution to load time and would mask
//       genuine link errors — which is exactly the "silent CPU fallback"
//       failure mode banned by FR-EX-08.
//
// Guarded on BuildTarget.iOS; other build targets fall through silently so
// this file is inert on desktop / Android / WebGL builds.
//
// NFR-RL-03 (App Store forbids dlopen of custom dylibs) is satisfied
// upstream by using the static libvokra.a with [DllImport("__Internal")];
// this hook only guarantees the link phase resolves correctly.

#if UNITY_IOS
using System.IO;
using UnityEditor;
using UnityEditor.Callbacks;
using UnityEditor.iOS.Xcode;
using UnityEngine;

namespace Vokra.Editor
{
    internal static class iOSPostProcessBuild
    {
        // Run late so other packages' PostProcessBuild hooks that mutate the
        // pbxproj (e.g. Firebase, third-party analytics) have already written
        // theirs before we validate OTHER_LDFLAGS.
        private const int CallbackOrder = 100;

        private const string StaticLibName = "libvokra.a";

        // Path inside the exported Xcode project where Unity places files
        // from Assets/Plugins/iOS/. This is Unity's fixed convention.
        private const string PluginsIosRelPath = "Libraries/com.vokra.unity/Plugins/iOS/";

        [PostProcessBuild(CallbackOrder)]
        public static void OnPostProcessBuild(BuildTarget target, string pathToBuiltProject)
        {
            if (target != BuildTarget.iOS)
            {
                return;
            }

            string pbxPath = PBXProject.GetPBXProjectPath(pathToBuiltProject);
            if (!File.Exists(pbxPath))
            {
                Debug.LogError("[Vokra] iOS PostProcessBuild: pbxproj not found at " + pbxPath);
                return;
            }

            PBXProject project = new PBXProject();
            project.ReadFromFile(pbxPath);

            // Unity 2019.3+ splits Unity-iPhone (app) and UnityFramework
            // (native code lives here). Native libs must be linked into the
            // framework target, not the app target.
            string frameworkTargetGuid = project.GetUnityFrameworkTargetGuid();
            string mainTargetGuid = project.GetUnityMainTargetGuid();

            // (a) Add libvokra.a to Link Binary With Libraries on UnityFramework.
            AddStaticLib(project, frameworkTargetGuid, pathToBuiltProject);

            // (b) Link Metal.framework and Accelerate.framework (non-weak) on
            //     the framework target where the native code lives.
            project.AddFrameworkToProject(frameworkTargetGuid, "Metal.framework", false);
            project.AddFrameworkToProject(frameworkTargetGuid, "Accelerate.framework", false);

            // (c) ENABLE_BITCODE=NO on both targets (Apple deprecated bitcode
            //     in Xcode 14; static archives without bitcode fail link on
            //     older Xcode if this is left ON).
            project.SetBuildProperty(frameworkTargetGuid, "ENABLE_BITCODE", "NO");
            project.SetBuildProperty(mainTargetGuid, "ENABLE_BITCODE", "NO");

            project.WriteToFile(pbxPath);

            // (d) After all pbxproj mutation (ours + any other package's) is
            //     flushed, re-read and assert OTHER_LDFLAGS on the framework
            //     target does not contain -undefined dynamic_lookup.
            VerifyNoUndefinedDynamicLookup(pbxPath, frameworkTargetGuid);
        }

        private static void AddStaticLib(PBXProject project, string targetGuid, string pathToBuiltProject)
        {
            // Unity copies Plugins/iOS/libvokra.a into the exported project
            // under Libraries/<package>/Plugins/iOS/. Look for it and add
            // explicitly to the link phase.
            string relPath = PluginsIosRelPath + StaticLibName;
            string absPath = Path.Combine(pathToBuiltProject, relPath);
            if (!File.Exists(absPath))
            {
                // Not fatal: the plugin .meta importer normally adds this,
                // and Unity's own copy step may place it under a different
                // subpath depending on package layout. Just warn.
                Debug.LogWarning("[Vokra] iOS PostProcessBuild: " + StaticLibName +
                                 " not found at " + relPath + "; relying on .meta importer.");
                return;
            }

            string fileGuid = project.AddFile(relPath, relPath, PBXSourceTree.Source);
            project.AddFileToBuild(targetGuid, fileGuid);
        }

        private static void VerifyNoUndefinedDynamicLookup(string pbxPath, string targetGuid)
        {
            // PBXProject exposes GetBuildPropertyForAnyConfig which returns
            // the merged value across Debug/Release/etc. If any config has
            // -undefined dynamic_lookup, we fail the build loudly.
            PBXProject verify = new PBXProject();
            verify.ReadFromFile(pbxPath);
            string ldflags = verify.GetBuildPropertyForAnyConfig(targetGuid, "OTHER_LDFLAGS");
            if (!string.IsNullOrEmpty(ldflags) && ldflags.Contains("-undefined dynamic_lookup"))
            {
                throw new BuildFailedException(
                    "[Vokra] OTHER_LDFLAGS contains '-undefined dynamic_lookup' on UnityFramework " +
                    "target. This defers symbol resolution to load time and masks link errors — " +
                    "banned by Vokra FR-EX-08 (no silent fallback). Remove it from your project or " +
                    "any conflicting PostProcessBuild hook.");
            }
        }
    }
}
#endif
