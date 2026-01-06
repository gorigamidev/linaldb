use crate::dsl::DslOutput;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize)]
pub struct Job {
    pub id: Uuid,
    pub command: String,
    pub status: JobStatus,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
    pub result: Option<DslOutput>,
    pub error: Option<String>,
    pub target_db: Option<String>,
}

pub struct JobManager {
    jobs: Arc<Mutex<HashMap<Uuid, Job>>>,
}

impl Default for JobManager {
    fn default() -> Self {
        Self::new()
    }
}

impl JobManager {
    pub fn new() -> Self {
        Self {
            jobs: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn create_job(&self, command: String, target_db: Option<String>) -> Uuid {
        let id = Uuid::new_v4();
        let job = Job {
            id,
            command,
            status: JobStatus::Pending,
            created_at: Utc::now(),
            started_at: None,
            finished_at: None,
            result: None,
            error: None,
            target_db,
        };
        self.jobs.lock().unwrap().insert(id, job);
        id
    }

    pub fn get_job(&self, id: Uuid) -> Option<Job> {
        self.jobs.lock().unwrap().get(&id).cloned()
    }

    pub fn list_jobs(&self) -> Vec<Job> {
        let jobs = self.jobs.lock().unwrap();
        jobs.values().cloned().collect()
    }

    pub fn update_job_status(&self, id: Uuid, status: JobStatus) {
        let mut jobs = self.jobs.lock().unwrap();
        if let Some(job) = jobs.get_mut(&id) {
            job.status = status.clone();
            if status == JobStatus::Running {
                job.started_at = Some(Utc::now());
            } else if status == JobStatus::Completed || status == JobStatus::Failed {
                job.finished_at = Some(Utc::now());
            }
        }
    }

    pub fn finish_job(&self, id: Uuid, result: Result<DslOutput, String>) {
        let mut jobs = self.jobs.lock().unwrap();
        if let Some(job) = jobs.get_mut(&id) {
            job.finished_at = Some(Utc::now());
            match result {
                Ok(output) => {
                    job.status = JobStatus::Completed;
                    job.result = Some(output);
                }
                Err(e) => {
                    job.status = JobStatus::Failed;
                    job.error = Some(e);
                }
            }
        }
    }
}
