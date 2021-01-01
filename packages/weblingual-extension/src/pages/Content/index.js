import { printLine } from './modules/print';
import { isContentParagraph, NERContent } from '../../utils';

console.log('Content script works!');
console.log('Must reload extension for modifications to take effect.');

printLine("Using the 'printLine' function from the Print Module");

let texts = []

let batch = 2;
function nextBatch(texts) {
    let thisBatch = texts.splice(0, batch);
    NERContent(thisBatch).then((results) => {
        results.forEach(({result, node, element}) => {
            element.replaceChild(document.createTextNode(result), node);
        });
        if (texts.length) nextBatch(texts);
    })
}

var uri = document.documentURI;
var elements = document.getElementsByTagName('*');

for (var i = 0; i < elements.length; i++) {
    var element = elements[i];

    for (var j = 0; j < element.childNodes.length; j++) {
        var node = element.childNodes[j];

        if (node.nodeType === 3 && node.parentNode.nodeName !== "SCRIPT") {
            var text = node.nodeValue;
            //var replacedText = text.replace(/[word or phrase to replace here]/gi, '[new word or phrase]');

            //if (replacedText !== text) {
            //    element.replaceChild(document.createTextNode(replacedText), node);
            //}
            if (isContentParagraph(text)) {
                texts.push({text: text, node: node, element: element})
            }
        }
    }
}

nextBatch(texts);

