# Batch Co-STORM

Run Co-STORM over many queries (e.g. from seo-geo cluster CSVs) with optional per-query injection documents. Supports local parallel runs and AWS Batch (Fargate).

## Overview

1. **Build manifest** – Turn cluster CSVs into a single manifest. `question_id` is `dataset_name + "_" + number` (e.g. `health_123`); `topic` is built from title and/or body.
2. **Single query** – Run one query through Co-STORM (warm start + N steps + report).
3. **Chunk** – Process a slice of the manifest (e.g. one AWS Batch job).
4. **Local parallel** – Process the full manifest with multiprocessing on one machine.

Outputs per query: `output_dir / <sanitized_question_id> / {report.md, instance_dump.json, log.json}`. Chunk/parallel runs also write a summary JSON (failed `question_id`s).

## Environment variables

- **LLM**: `OPENAI_API_TYPE` (openai | azure), `OPENAI_API_KEY` or `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION`. Loaded from `secrets.toml` if present (project root).
- **Retriever** (default: Google): `GOOGLE_SEARCH_API_KEY`, `GOOGLE_CSE_ID`. Alternatives: `BING_SEARCH_API_KEY`, `SERPER_API_KEY`, etc. (see `run_single_query --retriever`).
- **S3**: `boto3` and AWS credentials (env or IAM role). No extra env vars; use `aws configure` or Batch job role.

---

## 1. Build manifest

From project root (PYTHONPATH includes project root, or `pip install -e .`):

```bash
# Local cluster CSVs (e.g. cloned seo-geo)
python -m examples.batch.build_manifest \
  --input-dir /path/to/seo-geo/clustering_results/clusters \
  --topic-from title \
  --output ./manifest.csv

# Topic from title + body (truncated)
python -m examples.batch.build_manifest \
  --input-dir /path/to/clusters \
  --topic-from title_and_body \
  --body-max-chars 500 \
  --output ./manifest.csv

# From S3
python -m examples.batch.build_manifest \
  --input-s3 s3://YOUR_BUCKET/seo-geo/clusters/ \
  --topic-from title \
  --output ./manifest.csv

# Optional: injection mapping CSV (question_id, injection_doc_path, injection_doc_s3_uri)
python -m examples.batch.build_manifest \
  --input-dir /path/to/clusters \
  --injection-mapping ./injection_mapping.csv \
  --output ./manifest.csv

# Upload manifest to S3
python -m examples.batch.build_manifest \
  --input-dir /path/to/clusters \
  --output ./manifest.csv \
  --upload-s3 s3://YOUR_BUCKET/batch/manifest.csv
```

- **Duplicate question_id**: Default `first_wins`. Use `--duplicate fail_fast` to abort on duplicates.
- **Topic**: `--topic-from title` or `title_and_body`; `--body-max-chars` when using body.

---

## 2. Run a single query

```bash
python -m examples.batch.run_single_query \
  --question-id Q123 \
  --topic "Your topic here" \
  --output-dir ./results

# With optional injection doc (local or S3)
python -m examples.batch.run_single_query \
  --question-id Q123 \
  --topic "Your topic" \
  --injection-doc-path ./doc.json \
  --output-dir ./results

# From a manifest row (single-row CSV or JSON)
python -m examples.batch.run_single_query \
  --manifest-row ./one_row.csv \
  --output-dir ./results
```

Output: `./results/<sanitized_question_id>/report.md`, `instance_dump.json`, `log.json`. Skips if output exists unless `--no-skip-existing`.

---

## 3. Run a chunk (e.g. one Batch job)

```bash
# Chunk by index
python -m examples.batch.run_chunk \
  --manifest-s3 s3://YOUR_BUCKET/batch/manifest.csv \
  --chunk-id 0 \
  --chunk-size 100 \
  --output-dir ./chunk_results \
  --upload-s3 s3://YOUR_BUCKET/batch/results/chunk-0/

# By start/end index
python -m examples.batch.run_chunk \
  --manifest ./manifest.csv \
  --start-index 0 \
  --end-index 50 \
  --output-dir ./chunk_results
```

Writes `chunk_summary.json` (success_count, failure_count, failed list) and uploads to S3 if `--upload-s3` is set.

---

## 4. Run locally in parallel

```bash
python -m examples.batch.run_local_parallel \
  --manifest ./manifest.csv \
  --output-dir ./results \
  --workers 4 \
  --upload-s3 s3://YOUR_BUCKET/batch/results/
```

Same output layout; writes `parallel_summary.json`.

---

## 5. AWS Batch (Fargate)

The job definition (`infra/batch_job_def.json`) is for **Fargate**. The image entrypoint reads `AWS_BATCH_JOB_ARRAY_INDEX` and env vars so each array job runs the correct chunk and upload path. API keys are injected from a single JSON secret in Secrets Manager.

### AWS CLI setup (do this first)

1. **Install AWS CLI** (if needed): `brew install awscli` or [Install AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

2. **Configure credentials**:
   ```bash
   aws configure
   ```
   Enter: AWS Access Key ID, Secret Access Key, default region (e.g. `us-east-2`). Stored in `~/.aws/credentials` and `~/.aws/config`. Do not commit.

3. **Verify**: `aws sts get-caller-identity` (shows account ID and user).

### Full setup (one region for everything, e.g. us-east-2)

Set these before running any CLI commands below (use your profile and region):

```bash
export AWS_PROFILE=your-profile
export REGION=us-east-2
```

**1. Note account ID and region** – Top-right in Console: Account ID (12 digits), region (e.g. us-east-2).

**2. Create S3 bucket** – S3 → Create bucket. Name (e.g. `storm-geo-batch-<name>`), Block public access on, same region. Create prefix `batch/` (manifest at `batch/manifest.csv`, results under `batch/results/<run-id>/chunk-<index>/`; each submitted job gets its own `<run-id>` subfolder automatically).

**3. Create job role (container at runtime)** – IAM → Roles → Create role. Trusted entity: **Elastic Container Service** → **Elastic Container Service Task**. Create policy (JSON):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::YOUR-BUCKET-NAME",
        "arn:aws:s3:::YOUR-BUCKET-NAME/*"
      ]
    }
  ]
}
```

Name policy e.g. StormGeoBatchJobS3, attach to role. Role name e.g. StormGeoBatchJobRole. Copy **ARN** → `JOB_ROLE_ARN`.

**4. Create execution role (Fargate: ECR, CloudWatch)** – IAM → Roles → Create role. Trusted entity: **Elastic Container Service** → **Elastic Container Service Task**. Attach **AmazonECSTaskExecutionRolePolicy**. Role name e.g. StormGeoBatchExecutionRole. Copy **ARN** → `EXECUTION_ROLE_ARN`.

**5. Execution role trust** – IAM → StormGeoBatchExecutionRole → Trust relationships → Edit. Principal must include **ecs-tasks.amazonaws.com** (and optionally batch.amazonaws.com):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": ["batch.amazonaws.com", "ecs-tasks.amazonaws.com"]
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

**6. Create ECR repository** – ECR → Create repository, name `storm-geo-batch`. Note URI (e.g. `ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/storm-geo-batch`).

**7. Create Fargate compute environment** – Batch → Compute environments → Create. Type: **Fargate**. Name e.g. `storm-geo-fargate`. Service role: AWSServiceRoleForBatch. VPC/subnets: use **public subnets** if you want tasks to get a public IP (no VPC endpoints); or private subnets + VPC endpoints (see below). Create and wait for **VALID**.

**8. Create job queue** – Batch → Job queues → Create. Name e.g. `storm-geo-queue`. Compute environment: the Fargate env from step 7. Create.

**9. Create secret (Secrets Manager)** – One JSON secret with keys the container needs (e.g. OPENAI_API_KEY, GOOGLE_SEARCH_API_KEY, GOOGLE_CSE_ID). Console: Secrets Manager → Store new secret → Other type → Key/value, add keys, name e.g. `storm-geo-secrets`. Or CLI:

```bash
aws secretsmanager create-secret \
  --name storm-geo-secrets \
  --secret-string '{"OPENAI_API_KEY":"sk-...","GOOGLE_SEARCH_API_KEY":"...","GOOGLE_CSE_ID":"..."}' \
  --profile "$AWS_PROFILE" \
  --region "$REGION"
```

Get ARN:

```bash
aws secretsmanager describe-secret --secret-id storm-geo-secrets --profile "$AWS_PROFILE" --region "$REGION" --query ARN --output text
```

**10. Job role: allow reading the secret** – IAM → StormGeoBatchJobRole → Add permissions → Create inline policy (JSON):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "secretsmanager:GetSecretValue",
      "Resource": "arn:aws:secretsmanager:us-east-2:ACCOUNT_ID:secret:storm-geo-secrets-XXXXX"
    }
  ]
}
```

Use the full secret ARN from step 9.

**11. Job definition** – Copy `infra/batch_job_def.json.example` (do not commit the filled file). Replace: `<ACCOUNT_ID>`, `<REGION>`, `<BUCKET>`, `<JOB_ROLE_ARN>`, `<EXECUTION_ROLE_ARN>`, `<SECRET_ARN>` (the secret ARN from step 9). The example includes `networkConfiguration.assignPublicIp: ENABLED` so Fargate tasks get a public IP and can reach Secrets Manager/ECR when using public subnets. Register:

```bash
aws batch register-job-definition --cli-input-json file:///path/to/your-job-def.json --profile "$AWS_PROFILE" --region "$REGION"
```

**12. Build and push image** – From project root:

```bash
docker build -f infra/Dockerfile.batch -t storm-geo-batch .
aws ecr get-login-password --profile "$AWS_PROFILE" --region "$REGION" | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-2.amazonaws.com
docker tag storm-geo-batch:latest ACCOUNT_ID.dkr.ecr.us-east-2.amazonaws.com/storm-geo-batch:latest
docker push ACCOUNT_ID.dkr.ecr.us-east-2.amazonaws.com/storm-geo-batch:latest
```

**13. Upload manifest and run a test job**:

```bash
aws s3 cp manifest.csv s3://YOUR_BUCKET/batch/manifest.csv --profile "$AWS_PROFILE" --region "$REGION"
```

Submit one job (no array):

```bash
aws batch submit-job \
  --job-name storm-geo-test \
  --job-queue storm-geo-queue \
  --job-definition storm-geo-batch-chunk \
  --profile "$AWS_PROFILE" \
  --region "$REGION"
```

**14. See jobs**:

```bash
# List jobs on the queue
aws batch list-jobs --job-queue storm-geo-queue --profile "$AWS_PROFILE" --region "$REGION"

# By status
aws batch list-jobs --job-queue storm-geo-queue --job-status RUNNING --profile "$AWS_PROFILE" --region "$REGION"
aws batch list-jobs --job-queue storm-geo-queue --job-status SUCCEEDED --profile "$AWS_PROFILE" --region "$REGION"
aws batch list-jobs --job-queue storm-geo-queue --job-status FAILED --profile "$AWS_PROFILE" --region "$REGION"

# Describe one job (use jobId from submit or list)
aws batch describe-jobs --jobs JOB_ID --profile "$AWS_PROFILE" --region "$REGION"
```

Logs: CloudWatch → Log groups → `/aws/batch/job`.

**15. Array job** (e.g. 70 chunks for ~7000 rows at 100 per chunk):

```bash
aws batch submit-job \
  --job-name costorm-batch \
  --job-queue storm-geo-queue \
  --job-definition storm-geo-batch-chunk \
  --array-properties size=70 \
  --profile "$AWS_PROFILE" \
  --region "$REGION"
```

### Test in the cloud: copy-paste commands

Use the same `REGION` and (if you use one) `AWS_PROFILE` as in your job definition. Replace `YOUR_BUCKET` and `your-job-queue` with your S3 bucket and Batch job queue name.

**One-time: confirm AWS and upload manifest**

```bash
export REGION=your-region
export AWS_PROFILE=your-profile
export BUCKET=your-bucket
export JOB_QUEUE=your-job-queue
export ACCOUNT_ID=your-account-id

# For me:
export REGION=us-east-2
export AWS_PROFILE=vitaly-aws
export BUCKET=storm-geo-batch-hal
export JOB_QUEUE=storm-geo-queue-public
export ACCOUNT_ID=122997907744

aws sts get-caller-identity --region "$REGION" --profile "$AWS_PROFILE"

aws s3 cp manifest.csv s3://$BUCKET/batch/manifest.csv --region "$REGION" --profile "$AWS_PROFILE"
```

**Run a single job (one chunk, good for testing)**

```bash
aws batch submit-job \
  --job-name storm-geo-test-$(date +%s) \
  --job-queue "$JOB_QUEUE" \
  --job-definition storm-geo-batch-chunk \
  --region "$REGION" --profile "$AWS_PROFILE"
```

Save the returned `jobId`; results go to `s3://YOUR_BUCKET/batch/results/<jobId>/chunk-0/` (each job gets its own subfolder automatically).

**Run an array (many chunks in parallel)**

```bash
aws batch submit-job \
  --job-name storm-geo-array-$(date +%s) \
  --job-queue "$JOB_QUEUE" \
  --job-definition storm-geo-batch-chunk \
  --array-properties size=5 \
  --region "$REGION" --profile "$AWS_PROFILE"
```

**Inspect jobs and results**

```bash
aws batch list-jobs --job-queue $JOB_QUEUE --job-status RUNNING --region "$REGION" --profile "$AWS_PROFILE"
aws batch list-jobs --job-queue $JOB_QUEUE --job-status SUCCEEDED --region "$REGION" --profile "$AWS_PROFILE"

aws batch describe-jobs --jobs $JOB_ID --region "$REGION" --profile "$AWS_PROFILE"

aws s3 ls s3://$BUCKET/batch/results/ --region "$REGION" --profile "$AWS_PROFILE"
# List chunks for a specific run (run ID = job ID from submit, or BATCH_RUN_PREFIX if set)
aws s3 ls s3://$BUCKET/batch/results/RUN_ID/ --region "$REGION" --profile "$AWS_PROFILE"
aws s3 ls s3://$BUCKET/batch/results/RUN_ID/chunk-0/ --region "$REGION" --profile "$AWS_PROFILE"
```

**View logs (CloudWatch)**

From `describe-jobs` output, use the job’s `container.logStreamName` (and the log group from your compute environment, often `/aws/batch/job`):

```bash
aws logs get-log-events \
  --log-group-name /aws/batch/job \
  --log-stream-name "LOG_STREAM_NAME_FROM_DESCRIBE_JOBS" \
  --region "$REGION" --profile "$AWS_PROFILE"
```

Or in the console: Batch → Jobs → job name → Log stream.

**If something isn’t set up**

- **Job queue name** – List queues: `aws batch describe-job-queues --region "$REGION" --query 'jobQueues[*].jobQueueName' --profile "$AWS_PROFILE"`
- **ECR and image** – Build and push so the job definition’s image URI exists:
  ```bash
  docker build -f infra/Dockerfile.batch -t storm-geo-batch .
  aws ecr get-login-password --region "$REGION" --profile "$AWS_PROFILE" | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
  docker tag storm-geo-batch:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/storm-geo-batch:latest
  docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/storm-geo-batch:latest
  ```
- **Secret** – The job definition references a Secrets Manager secret (e.g. `storm-geo-secrets`). Ensure that secret exists in the same region and contains the keys your app expects (e.g. `OPENAI_API_KEY`, `GOOGLE_SEARCH_API_KEY`, `GOOGLE_CSE_ID`).

### Private subnets: VPC endpoints (no NAT, no public IP)

If your Fargate compute environment uses **private subnets**, tasks need a path to Secrets Manager and ECR. Use VPC endpoints so traffic stays inside AWS.

Set variables (use your profile, region, VPC and private subnet IDs):

```bash
export AWS_PROFILE=your-profile
export REGION=us-east-2
export VPC_ID=vpc-xxxxxxxx
export SUBNET_1=subnet-aaaaaaaa
export SUBNET_2=subnet-bbbbbbbb
```

Create security group for endpoints:

```bash
VPCE_SG_ID=$(aws ec2 create-security-group \
  --group-name storm-geo-vpce-sg \
  --description "VPC endpoints for Batch Fargate" \
  --vpc-id "$VPC_ID" \
  --profile "$AWS_PROFILE" \
  --region "$REGION" \
  --output text --query GroupId)
export VPC_CIDR=172.31.0.0/16
aws ec2 authorize-security-group-ingress \
  --group-id "$VPCE_SG_ID" \
  --protocol tcp --port 443 --cidr "$VPC_CIDR" \
  --profile "$AWS_PROFILE" \
  --region "$REGION"
```

Create interface endpoints (Secrets Manager + ECR):

```bash
aws ec2 create-vpc-endpoint --vpc-id "$VPC_ID" --vpc-endpoint-type Interface \
  --service-name "com.amazonaws.${REGION}.secretsmanager" \
  --subnet-ids "$SUBNET_1" "$SUBNET_2" --security-group-ids "$VPCE_SG_ID" \
  --profile "$AWS_PROFILE" \
  --region "$REGION" --no-cli-pager

aws ec2 create-vpc-endpoint --vpc-id "$VPC_ID" --vpc-endpoint-type Interface \
  --service-name "com.amazonaws.${REGION}.ecr.api" \
  --subnet-ids "$SUBNET_1" "$SUBNET_2" --security-group-ids "$VPCE_SG_ID" \
  --profile "$AWS_PROFILE" \
  --region "$REGION" --no-cli-pager

aws ec2 create-vpc-endpoint --vpc-id "$VPC_ID" --vpc-endpoint-type Interface \
  --service-name "com.amazonaws.${REGION}.ecr.dkr" \
  --subnet-ids "$SUBNET_1" "$SUBNET_2" --security-group-ids "$VPCE_SG_ID" \
  --profile "$AWS_PROFILE" \
  --region "$REGION" --no-cli-pager
```

S3 gateway endpoint (so tasks can read/write S3 without NAT):

```bash
ROUTE_TABLE_ID=$(aws ec2 describe-route-tables \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=association.main,Values=true" \
  --profile "$AWS_PROFILE" \
  --region "$REGION" --query 'RouteTables[0].RouteTableId' --output text)
aws ec2 create-vpc-endpoint --vpc-id "$VPC_ID" --vpc-endpoint-type Gateway \
  --service-name "com.amazonaws.${REGION}.s3" \
  --route-table-ids "$ROUTE_TABLE_ID" \
  --profile "$AWS_PROFILE" \
  --region "$REGION" --no-cli-pager
```

Then re-run your Batch job; no change to the job definition.

### Public subnets (simpler: tasks get a public IP)

Use **public subnets** so Fargate tasks get a public IP and can reach Secrets Manager, ECR, and S3 over the internet. No VPC endpoints needed. Run these after you have the job role, execution role, ECR repo, job definition, and image built (steps 3–6, 9–12 above). Use a **new** compute environment and job queue so you can keep your existing private-subnet setup if you have one.

**1. Set variables**

```bash
export AWS_PROFILE=vitaly-aws
export REGION=us-east-2
```

**2. Get default VPC and its public subnets**

```bash
export VPC_ID=$(aws ec2 describe-vpcs \
  --filters "Name=is-default,Values=true" \
  --profile "$AWS_PROFILE" \
  --region "$REGION" \
  --query 'Vpcs[0].VpcId' --output text)

export SUBNET_IDS=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" \
  --profile "$AWS_PROFILE" \
  --region "$REGION" \
  --query 'Subnets[*].SubnetId' --output text)

echo "VPC_ID=$VPC_ID"
echo "SUBNETS=$SUBNET_IDS"
```

**3. Get default security group**

```bash
export SECURITY_GROUP=$(aws ec2 describe-security-groups \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=group-name,Values=default" \
  --profile "$AWS_PROFILE" \
  --region "$REGION" \
  --query 'SecurityGroups[0].GroupId' --output text)
echo "SECURITY_GROUP=$SECURITY_GROUP"
```

**4. Get Batch service role ARN**

```bash
export BATCH_SERVICE_ROLE_ARN=$(aws iam get-role \
  --role-name AWSServiceRoleForBatch \
  --profile "$AWS_PROFILE" \
  --query 'Role.Arn' --output text 2>/dev/null || echo "arn:aws:iam::YOUR_ACCOUNT_ID:role/AWSServiceRoleForBatch")
echo "BATCH_SERVICE_ROLE_ARN=$BATCH_SERVICE_ROLE_ARN"
```

If that role does not exist, create it in the Console (Batch → Create compute environment usually offers to create it), then set `BATCH_SERVICE_ROLE_ARN` to its ARN.

**5. Create Fargate compute environment (public subnets)**

```bash
SUBNET_JSON=$(echo $SUBNET_IDS | awk '{for(i=1;i<=NF;i++)printf "\"%s\"%s", $i, (i<NF?", ":"")}')
export COMPUTE_ENV_NAME=storm-geo-fargate-public

aws batch create-compute-environment \
  --compute-environment-name "$COMPUTE_ENV_NAME" \
  --type MANAGED \
  --state ENABLED \
  --service-role "$BATCH_SERVICE_ROLE_ARN" \
  --compute-resources "type=FARGATE,maxvCpus=256,subnets=[$SUBNET_JSON],securityGroupIds=[$SECURITY_GROUP]" \
  --profile "$AWS_PROFILE" \
  --region "$REGION" \
  --no-cli-pager
```

**6. Wait until compute environment is VALID**

```bash
aws batch describe-compute-environments \
  --compute-environments "$COMPUTE_ENV_NAME" \
  --profile "$AWS_PROFILE" \
  --region "$REGION" \
  --query 'computeEnvironments[0].status' --output text
```

Repeat until it returns `VALID`.

**7. Create job queue for this compute environment**

```bash
export JOB_QUEUE=storm-geo-queue-public

aws batch create-job-queue \
  --job-queue-name "$JOB_QUEUE" \
  --state ENABLED \
  --priority 1 \
  --compute-environment-order "order=1,computeEnvironment=$COMPUTE_ENV_NAME" \
  --profile "$AWS_PROFILE" \
  --region "$REGION" \
  --no-cli-pager
```

**8. Submit a job to the new queue**

```bash
aws batch submit-job \
  --job-name storm-geo-test \
  --job-queue "$JOB_QUEUE" \
  --job-definition storm-geo-batch-chunk \
  --profile "$AWS_PROFILE" \
  --region "$REGION"
  # --container-overrides '{"environment":[{"name":"envvar","value":"123"}]}'
```

Use the same job definition and image; only the queue points at the new compute environment. Tasks run in public subnets with a public IP and can reach Secrets Manager, ECR, and S3.

---

## Results in S3

Each submitted job gets a unique run folder so runs do not overwrite each other. Run ID is the Batch job ID (for array jobs, the part before `:array-index`) unless overridden. Layout:

- `s3://BUCKET/batch/results/<run-id>/chunk-<index>/` – per chunk: `question_id/` dirs and `chunk_summary.json`.

To use a custom run prefix (e.g. timestamp or name), pass it when submitting:

```bash
aws batch submit-job \
  --job-name storm-geo-test \
  --job-queue your-job-queue \
  --job-definition storm-geo-batch-chunk \
  --container-overrides '{"environment":[{"name":"BATCH_RUN_PREFIX","value":"run-20250202-143000"}]}' \
  --region "$REGION"
```

For array jobs, set `BATCH_RUN_PREFIX` the same way so all chunks go under one run folder.

---

## Code sharing

- **FirstRetrievalInjector**: `examples/batch/_injector.py` (inject docs into first retrieval).
- **create_information_from_dict**: from `examples.manual_examples.manual_document_helper`; run from project root with PYTHONPATH so `examples` resolves.

---

## question_id and duplicate policy

Manifest `question_id` is **dataset_name + "_" + number** (e.g. `health_123` from `health_clusters.csv`). The manifest keeps `original_question_id` and `dataset`. Injection mapping CSV uses the new `question_id`. Duplicates: `--duplicate first_wins` (default) or `fail_fast`.

---

## Rate limits

With many Batch jobs in parallel, LLM and search APIs may rate-limit. Options: reduce array size or increase chunk size; add a per-job delay in the entrypoint; rely on client retries. Tune `--chunk-size` and array size accordingly.
