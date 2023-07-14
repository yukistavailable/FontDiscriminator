"use client";
import { useState, useRef } from "react";
export default function Home() {
  const [tasks, setTasks] = useState([]);
  const inputRef = useRef<HTMLInputElement>(null);
  function addTask() {
    const task: string = inputRef.current?.value ?? "";
    setTasks(tasks.concat(task));
  }

  function deleteTask() {
    const newTasks: string[] = [];
    for (let i = 0; i < tasks.length; i++) {
      const chk: HTMLInputElement | null = document.querySelector(
        `input[name=chk${i.toString()}]`
      );
      if (chk?.checked === false) {
        newTasks.push(tasks[i]);
      }
      // set the checked property to false
      if (chk) {
        chk.checked = false;
      }
    } 
    setTasks(newTasks);
  }


  function InputField() {
    return (
      <>
        <div>
          Task: <input id="task" type="text" ref={inputRef} />
        </div>
        <button
          className="add_btn"
          onClick={addTask}
        >
          Add
        </button>
      </>
    );
  }
  return (
    <main className="main_div">
      Todo List
      <InputField />
      <div>
        {tasks.map((e, i) => (
          // eslint-disable-next-line react/jsx-key
          <label className="check_lb">
            <input type="checkbox" name={`chk${i.toString()}`} />
            {e}
          </label>
        ))}
        <button
          className="add_btn"
          onClick={deleteTask}
        >
          Done
        </button>
      </div>
    </main>
  );
}
