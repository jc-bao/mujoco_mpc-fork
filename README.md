# Notes on adding new tasks to mjpc

## Add new task

Step1: add new task folder under `mjpc/tasks`, including the following files: 

```
  assets
  gr1.cc (Define your reward function here)
  gr1.h (Define your reward function here)
󰗀  gr1.xml (Put your robot model here, remember to design sensor to get states)
󰗀  task_flat.xml (Define your task here)
󰗀  task_hill.xml
```

Step2: in `mjpc/CMakeLists.txt` can add `add_library` for your new task.

```
add_library(
    libmjpc STATIC

    # TODO: add your new task here
    tasks/gr1/gr1.cc
    tasks/gr1/gr1.h

    states/state.cc
    states/state.h
    agent.cc
...
)
```

Step3: in `mjpc/tasks/tasks.cc`, `include`, `make_shared` for your new task.

```
#include "gr1/gr1.h"
    std::vector<std::shared_ptr<Task>> GetTasks()
    {
        return {
            ...
            std::make_shared<GR1Flat>(),
        };
    }
...
```