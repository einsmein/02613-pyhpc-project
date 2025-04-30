Project code structure
- `simulate.py` module loads data, simulates, prints results. 
To optimize the code, we have to improve `jacobi` and `simulate` function to begin with.
We might be able to gain speedup with how data is loaded, but let's see.

- `visualize.py` module contains a function to visualize
an initial condition and a mask for a building.
Might be useful for debugging purposes.
To use it, run `python visualize.py <building_id>.


Git basic
- Set up the repository
    - Clone this repository with git clone
    - Create your branch
        `git checkout -b <branch_name>`
    - Push your branch to github
        `git push -u origin HEAD`
- Work on the project
    - After making changes, stage files to commit
        `git add <file1> <file2>`
    - Commit the changes and push to github
        `git commit -m "<add comment here>"; git push`
    - You can always check states of your repo (which branch you're on, which files are changed, which are staged, etc.)
        `git status`
- Other commands
    - To check out different branch
        `git checkout <branch_name>`

    