/**
 * commands/docs.ts - document retrieval and listing commands
 * (get / multi-get / ls).
 *
 * Currently hosts only the path-rendering helpers; the command functions
 * land here in batch 4 of the qmd.ts split (docs/specs/qmd-cli-file-split.md).
 */
import { realpathSync } from "fs";
import { relative as relativePath } from "path";

function renderFullPath(absolutePath: string, cwd: string = process.cwd()): string {
  let real: string;
  try { real = realpathSync(absolutePath); } catch { real = absolutePath; }
  const cwdReal = (() => { try { return realpathSync(cwd); } catch { return cwd; } })();
  if (real === cwdReal) return "./";
  if (real.startsWith(cwdReal + "/")) {
    const rel = relativePath(cwdReal, real);
    if (rel && !rel.startsWith("..")) return `./${rel}`;
  }
  return real;
}

// Compute unique display path for a document
// Always include at least parent folder + filename, add more parent dirs until unique
function computeDisplayPath(
  filepath: string,
  collectionPath: string,
  existingPaths: Set<string>
): string {
  // Get path relative to collection (include collection dir name)
  const collectionDir = collectionPath.replace(/\/$/, '');
  const collectionName = collectionDir.split('/').pop() || '';

  let relativePath: string;
  if (filepath.startsWith(collectionDir + '/')) {
    // filepath is under collection: use collection name + relative path
    relativePath = collectionName + filepath.slice(collectionDir.length);
  } else {
    // Fallback: just use the filepath
    relativePath = filepath;
  }

  const parts = relativePath.split('/').filter(p => p.length > 0);

  // Always include at least parent folder + filename (minimum 2 parts if available)
  // Then add more parent dirs until unique
  const minParts = Math.min(2, parts.length);
  for (let i = parts.length - minParts; i >= 0; i--) {
    const candidate = parts.slice(i).join('/');
    if (!existingPaths.has(candidate)) {
      return candidate;
    }
  }

  // Absolute fallback: use full path (should be unique)
  return filepath;
}

export {
  renderFullPath,
};
