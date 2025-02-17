# Instructions

## How to add new environments

1. Create task folder in `mjpc/tasks`
2. Include `task.cc`, `task.h`, `model.xml`, `task.xml`
3. Edit `mjpc/CMakeLists.txt`, add source file like `tasks/g1/tracking/tracking.cc`, `tasks/g1/tracking/tracking.h`
4. Edit `mjpc/tasks/tasks.cc`, add header file like `tasks/g1/tracking/tracking.h` and new task pointer like `std::make_shared<g1::Tracking>(),`.

## How to import new motion

## How to set motion in the real time 