output "external_ip" {
  description = "External IP address of the toxtransformer instance"
  value       = google_compute_address.toxtransformer.address
}

output "instance_name" {
  description = "Name of the GCE instance"
  value       = google_compute_instance.toxtransformer.name
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "gcloud compute ssh ${google_compute_instance.toxtransformer.name} --zone=${var.zone}"
}

output "image_uri" {
  description = "Docker image URI pushed to Artifact Registry"
  value       = local.image_uri
}

output "api_url" {
  description = "API endpoint URL"
  value       = "http://${google_compute_address.toxtransformer.address}:6515"
}

output "labels" {
  description = "Resource labels for traceability"
  value       = local.common_labels
}

output "load_balancer_ip" {
  description = "Load balancer IP address (point toxtransformer.toxindex.com DNS A record here)"
  value       = google_compute_global_address.lb.address
}

output "streamlit_url" {
  description = "Streamlit UI URL (via load balancer)"
  value       = "https://toxtransformer.toxindex.com"
}

output "cloud_run_url" {
  description = "Direct Cloud Run URL (for debugging)"
  value       = google_cloud_run_service.streamlit.status[0].url
}
