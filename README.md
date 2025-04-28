You can find our data at this link https://archive.org/download/stackexchange/.

Code is coming soon

Process data into three parts: teams, experts, and tasks, and build index among them


1.You can use E2t.py to embed the team

2.Use ExpertEmbeding.py to get the embedding of experts and tasks

3.Use model_RI.py to train the task-relevance model which can learn the relevance between new task and historical tasks to find K similar tasks.

4.Forming the teams and get the results based on the embedding of new task.
