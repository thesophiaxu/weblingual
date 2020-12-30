import { Button, ListItem, MenuItem, Select, TextField, Typography } from '@material-ui/core';
import React, { ChangeEvent } from 'react';
import { sendNLPRequest } from '../../utils';
import './Options.css';
import ReactJson from 'react-json-view'

interface Props {
  title: string;
}

function SendRequest () {
  const [task, setTask] = React.useState('')
  const [text, setText] = React.useState('')
  const [result, setResult] = React.useState({})
  return <div>
    <TextField id="standard-basic" label="Text" value={text} onChange={ev => {setText(ev.target.value as string)}}/>
    <Select
        labelId="demo-simple-select-label"
        id="demo-simple-select"
        value={task}
        onChange={ev => {setTask(ev.target.value as string)}}
      >
        <MenuItem value={"ner"}>Entity Recognition</MenuItem>
        <MenuItem value={"sentiment"}>Sentiment Analysis</MenuItem>
      </Select>
    <Button onClick={() => {sendNLPRequest(task, [text]).then(setResult as any)}}>Send</Button>
    <br/>
    <ReactJson src={result as any} /> 
  </div>
}

const Options: React.FC<Props> = ({ title }: Props) => {
  

  return <div className="OptionsContainer">
    <Typography variant="h3">WebLingual Options</Typography>
    <Typography variant="h4">Send request</Typography>
    <ListItem>
      <SendRequest/>
    </ListItem>
  </div>;
};

export default Options;
