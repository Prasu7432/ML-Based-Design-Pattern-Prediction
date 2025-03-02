package org.eclipse.cdt.core.resources;

import java.util.Map;
import java.util.Map.Entry;
import org.eclipse.cdt.core.CCorePlugin;
import org.eclipse.cdt.core.IMarkerGenerator;
import org.eclipse.cdt.core.ProblemMarkerInfo;
import org.eclipse.core.resources.IMarker;
import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.IResource;
import org.eclipse.core.resources.IncrementalProjectBuilder;
import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.IProgressMonitor;

public class MLDesignPatternBuilder extends IncrementalProjectBuilder implements IMarkerGenerator {

    private static final boolean DEBUG_EVENTS = true;
    private MLModel mlModel = new MLModel(); // Assume MLModel is a class that makes predictions

    @Override
    protected IProject[] build(int kind, Map<String, String> args, IProgressMonitor monitor) throws CoreException {
        IProject project = getProject();
        if (project == null) return null;
        
        analyzeProject(project);
        return null;
    }

    private void analyzeProject(IProject project) {
        try {
            IResource[] files = project.members();
            for (IResource file : files) {
                if (file.getFileExtension() != null && file.getFileExtension().equals("java")) {
                    String content = new String(file.getLocation().toFile().toPath().toString());
                    String prediction = mlModel.predict(content);
                    addMarker(file, prediction);
                }
            }
        } catch (Exception e) {
            CCorePlugin.log(e.getMessage());
        }
    }

    private void addMarker(IResource file, String pattern) {
        try {
            IMarker marker = file.createMarker(IMarker.PROBLEM);
            marker.setAttribute(IMarker.MESSAGE, "Detected Design Pattern: " + pattern);
            marker.setAttribute(IMarker.SEVERITY, IMarker.SEVERITY_WARNING);
        } catch (CoreException e) {
            CCorePlugin.log(e.getStatus());
        }
    }

    @Override
    protected void clean(IProgressMonitor monitor) throws CoreException {
        getProject().deleteMarkers(IMarker.PROBLEM, true, IResource.DEPTH_INFINITE);
        if (DEBUG_EVENTS) System.out.println("Cleaned all markers.");
    }
}