import fs from "node:fs";
import path from "node:path";

const names = ["react", "react-dom", "scheduler", "lucide-react"];
const notices = names.map((name) => {
  const directory = path.join("node_modules", name);
  const metadata = JSON.parse(
    fs.readFileSync(path.join(directory, "package.json"), "utf8"),
  );
  return `${name} ${metadata.version}\n${"=".repeat(72)}\n${fs.readFileSync(path.join(directory, "LICENSE"), "utf8")}`;
});
fs.writeFileSync("dist/third-party-licenses.txt", notices.join("\n\n"));
