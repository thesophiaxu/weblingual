export function sendNLPRequest(task: string, text: string[]) {
    return new Promise((resolve, reject) => {
        fetch('http://localhost:22222/tasks/' + task, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
                // 'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: JSON.stringify({texts: text})
        }).then(async (response) => {
            let j_text = await response.json()
            let j = JSON.parse(j_text.replaceAll("'", "\""))
            if(j) {
                resolve(j)
            } else {
                reject(response)
            }
        })
    })
}

/**
 * Uses simple heuristics to determine whether a paragraph is a content.
 * @param content 
 */
export function isContentParagraph(content: string) {
    const startsWithSpaces = content.startsWith('    ')
    const startsWithTab = content.startsWith('\t')
    const notLongEnough = content.replaceAll(' ', '').length < 20
    const tooManySpaces = content.includes('        ')

    if (startsWithSpaces || startsWithTab || notLongEnough || tooManySpaces) return false;
    else return true;
}

export function NERContent(texts: any[]) {
    return new Promise((resolve, reject) => {
        let res: any[] = []
        let textsOnly: string[] = texts.reduce((curr, next: any) => {curr.push(next.text); return curr;}, []);
        sendNLPRequest('ner', textsOnly).then((results: any) => {
            results.result.forEach((entry: any, idx: number) => {
                let replaced = textsOnly[idx];
                entry.forEach((entry: any) => {
                    replaced = replaced.replace(entry.word, `[[${entry.word}]]`)
                });
                res.push({result: replaced, node: texts[idx].node, element: texts[idx].element})
            })
            resolve(res);
        })
    })
}