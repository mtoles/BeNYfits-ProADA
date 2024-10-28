const express = require('express');
const path = require('path');
const app = express();

function findLatestDir() {
    const dirs = fs.readdirSync('../results');
    const experiments = dirs.filter(dir => fs.statSync(`../results/${dir}`).isDirectory());
    // get all subdirs in each experiment dir
    const subdirs = experiments.flatMap(experiment => {
        return fs.readdirSync(`../results/${experiment}`).filter(subdir => fs.statSync(`../results/${experiment}/${subdir}`).isDirectory());
    });
    // sort subdirs alphabetically
    const sortedSubdirs = subdirs.sort();
    // find the last sub dir in the sorted subdirs array
    const lastSubdir = sortedSubdirs[sortedSubdirs.length - 1];
    // get the actual file so we can send it to displayJsonFile
    const outputPath = `../results/${experiments[0]}/${lastSubdir}/history.jsonl`;
    console.log(`outputPath: ${outputPath}`);
    return outputPath;
}

// Serve a static file from the server
app.get('/latest', (req, res) => {
//   const filePath = path.join(__dirname, 'files', 'example.pdf'); // Path to the file
    console.log('fetching latest file (server)');
    const filePath = findLatestDir();
    res.download(filePath, (err) => {
        if (err) {
            res.status(500).send('Error downloading the file');
        }
    });
});

// app.get('/download', (req, res) => {
//     const filePath = path.join(__dirname, 'files', 'example.pdf'); // Path to the file
//     res.download(filePath, (err) => {
//       if (err) {
//         res.status(500).send('Error downloading the file');
//       }
//     });
//   });
  

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
