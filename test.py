import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta

# Parameters
num_tasks = 100  # Total number of tasks
num_team_members = 10  # Total number of team members
skills_pool = ["Python", "Data Analysis", "Project Management", "Machine Learning", "Design", "Testing", "Frontend", "Backend", "DevOps"]

# Generate Team Member Data
team_members = []
for i in range(num_team_members):
    member_id = f"TM{i+1}"
    member_skills = random.sample(skills_pool, k=random.randint(1, 4))
    current_workload = round(random.uniform(0.2, 1.0), 2)  # workload in a range from 0.2 to 1.0 (20%-100%)
    team_members.append({
        "Team Member ID": member_id,
        "Team Member Skills": member_skills,
        "Current Workload": current_workload
    })

# Convert to DataFrame
team_members_df = pd.DataFrame(team_members)

# Generate Task Data
tasks = []
for i in range(num_tasks):
    task_id = f"TASK{i+1}"
    task_description = f"Complete task {i+1} related to project work."
    required_skills = random.sample(skills_pool, k=random.randint(1, 3))
    deadline = datetime.now() + timedelta(days=random.randint(1, 30))  # random deadline within the next 30 days
    priority = random.choice(["High", "Medium", "Low"])
    
    # Define task dependencies randomly
    dependencies = random.sample([f"TASK{j+1}" for j in range(num_tasks) if j != i], k=random.randint(0, 2))

    tasks.append({
        "Task ID": task_id,
        "Task Description": task_description,
        "Required Skills": required_skills,
        "Deadline": deadline.strftime("%Y-%m-%d"),
        "Priority": priority,
        "Task Dependencies": dependencies
    })

# Convert to DataFrame
tasks_df = pd.DataFrame(tasks)

# Merge Task and Team Member Data for Skill Matching
# Randomly assign each task to a team member with matching skills
task_distribution = []
for _, task in tasks_df.iterrows():
    # Filter team members with at least one matching skill and workload < 1.0
    suitable_members = team_members_df[team_members_df['Team Member Skills'].apply(lambda skills: any(skill in skills for skill in task['Required Skills']))]
    suitable_members = suitable_members[suitable_members['Current Workload'] < 1.0]
    
    # Randomly assign a suitable member if any are available
    if not suitable_members.empty:
        assigned_member = suitable_members.sample(1).iloc[0]
        task_distribution.append({
            "Task ID": task['Task ID'],
            "Task Description": task['Task Description'],
            "Required Skills": task['Required Skills'],
            "Deadline": task['Deadline'],
            "Priority": task['Priority'],
            "Task Dependencies": task['Task Dependencies'],
            "Assigned Team Member ID": assigned_member['Team Member ID'],
            "Assigned Team Member Skills": assigned_member['Team Member Skills'],
            "Team Member Workload": assigned_member['Current Workload']
        })
    else:
        # If no team member is available, leave assignment blank
        task_distribution.append({
            "Task ID": task['Task ID'],
            "Task Description": task['Task Description'],
            "Required Skills": task['Required Skills'],
            "Deadline": task['Deadline'],
            "Priority": task['Priority'],
            "Task Dependencies": task['Task Dependencies'],
            "Assigned Team Member ID": None,
            "Assigned Team Member Skills": None,
            "Team Member Workload": None
        })

# Convert to DataFrame
task_distribution_df = pd.DataFrame(task_distribution)

# Save the generated dataset to a CSV file
csv_file_path = 'C:\Users\Venom\My Files\Course\AI\NLP\Project\AI-driven task distribution expert model'
task_distribution_df.to_csv(csv_file_path, index=False)

# Display the path of the saved file
csv_file_path
