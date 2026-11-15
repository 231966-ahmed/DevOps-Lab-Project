
import { Component, signal, inject } from '@angular/core';
import { KeyValuePipe } from '@angular/common';
import { HttpClient } from '@angular/common/http';

interface PredictionResult {
  row: number;
  prediction: string;
}

interface PredictionResponse {
  success: boolean;
  model_used: string;
  total_records: number;
  prediction_counts: Record<string, number>;
  results: PredictionResult[];
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [KeyValuePipe],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {

  private http = inject(HttpClient);

  protected readonly title = 'CIC-IoT2023 Intrusion Detection';

  protected selectedModel = signal('random_forest');
  protected selectedFile = signal<File | null>(null);

  protected loading = signal(false);
  protected errorMessage = signal('');
  protected predictionData = signal<PredictionResponse | null>(null);

  protected onModelChange(event: Event): void {
    const value = (event.target as HTMLSelectElement).value;
    this.selectedModel.set(value);
  }

  protected onFileSelected(event: Event): void {
    const file = (event.target as HTMLInputElement).files?.[0] ?? null;

    this.selectedFile.set(file);
    this.errorMessage.set('');
    this.predictionData.set(null);
  }

  protected runPrediction(): void {

    const file = this.selectedFile();

    if (!file) {
      this.errorMessage.set('Please select a CSV file first.');
      return;
    }

    const formData = new FormData();

    formData.append('file', file);
    formData.append('model', this.selectedModel());

    this.loading.set(true);
    this.errorMessage.set('');
    this.predictionData.set(null);

    this.http.post<PredictionResponse>(
      'http://127.0.0.1:5000/predict',
      formData
    ).subscribe({
      next: (response) => {
        this.predictionData.set(response);
        this.loading.set(false);
      },
      error: (error) => {
        console.error('Prediction error:', error);
        this.errorMessage.set(
          error.error?.error || 'Unable to connect to the prediction API.'
        );
        this.loading.set(false);
      }
    });
  }
}
