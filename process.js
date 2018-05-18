//青空文庫からテキストをダウンロードし， nkf -w --overwrite sample.txt でutfにした．
//次に全角スペースと「と」をテキストエディタで削除した．

var fs = require("fs");
var readline = require("readline");

var stream = fs.createReadStream(process.argv[2], "utf8");

var reader = readline.createInterface({ input: stream });
reader.on("line", (data) => {
  var d1 = data.trim().replace(/《.*?》/g,"");
  var sents = d1.split("。");
  for (var i=0;i<sents.length;i++){
    if (sents[i].length <= 1) continue;
    var line = sents[i]+","+process.argv[4]+"\n";
    fs.appendFile(process.argv[3],line);
    console.log(line);
  }
});
