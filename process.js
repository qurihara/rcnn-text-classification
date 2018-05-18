var fs = require("fs");
var readline = require("readline");

var stream = fs.createReadStream(process.argv[2], "utf8");

var reader = readline.createInterface({ input: stream });
reader.on("line", (data) => {
  var d1 = data.trim();
  var sents = d1.split("。");
  for (var i=0;i<sents.length;i++){
    console.log(sents[i]);
  }
});
