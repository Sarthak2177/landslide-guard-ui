import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Checkbox } from '@/components/ui/checkbox';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { AlertTriangle, CheckCircle, Cloud, Mountain, Layers, Leaf, Activity } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface PredictionData {
  // Weather Data
  rainfall: string;
  temperature: string;
  humidity: string;
  rainfall3Day: string;
  rainfall7Day: string;

  // Topographical Data
  slopeAngle: string;
  aspect: string;
  elevation: string;

  // Soil Properties
  soilSaturation: string;
  soilPh: string;
  clayContent: string;
  sandContent: string;
  siltContent: string;
  soilErosionRate: string;
  poreWaterPressure: string;
  soilMoistureContent: string;
  soilTemperature: string;
  soilStrain: string;
  tdrReflectionIndex: string;

  // Environmental Data
  vegetationCover: string;
  ndviIndex: string;
  landUse: string;
  proximityToWater: string;
  distanceToRoad: string;

  // Geological Data
  earthquakeActivity: string;
  historicalLandslideCount: string;
  microseismicActivity: string;
  acousticEmission: string;
  soilType: string[];
}

interface PredictionResult {
  risk: 'high' | 'low';
  confidence: number;
  message: string;
}

const PredictionForm = () => {
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [formData, setFormData] = useState<PredictionData>({
    rainfall: '',
    temperature: '',
    humidity: '',
    rainfall3Day: '',
    rainfall7Day: '',
    slopeAngle: '',
    aspect: '',
    elevation: '',
    soilSaturation: '',
    soilPh: '',
    clayContent: '',
    sandContent: '',
    siltContent: '',
    soilErosionRate: '',
    poreWaterPressure: '',
    soilMoistureContent: '',
    soilTemperature: '',
    soilStrain: '',
    tdrReflectionIndex: '',
    vegetationCover: '',
    ndviIndex: '',
    landUse: '',
    proximityToWater: '',
    distanceToRoad: '',
    earthquakeActivity: '',
    historicalLandslideCount: '',
    microseismicActivity: '',
    acousticEmission: '',
    soilType: [],
  });

  const handleInputChange = (field: keyof PredictionData, value: string) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSoilTypeChange = (soilType: string, checked: boolean) => {
    setFormData(prev => ({
      ...prev,
      soilType: checked 
        ? [...prev.soilType, soilType]
        : prev.soilType.filter(type => type !== soilType)
    }));
  };

  const handlePredict = async () => {
    setIsLoading(true);
    
    // Simulate API call
    try {
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Mock prediction result
      const mockResult: PredictionResult = {
        risk: Math.random() > 0.5 ? 'high' : 'low',
        confidence: Math.floor(Math.random() * 30) + 70,
        message: Math.random() > 0.5 ? 'Landslide Likely' : 'Landslide Unlikely'
      };
      
      setResult(mockResult);
      
      toast({
        title: "Prediction Complete",
        description: `Risk assessment: ${mockResult.risk} (${mockResult.confidence}% confidence)`,
      });
    } catch (error) {
      toast({
        title: "Prediction Failed",
        description: "Unable to process prediction. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background p-4">
      <div className="mx-auto max-w-4xl">
        <div className="mb-8 text-center">
          <h1 className="mb-2 text-3xl font-bold tracking-tight">Landslide Prediction System</h1>
          <p className="text-muted-foreground">
            Enter environmental parameters to assess landslide risk
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <ScrollArea className="h-[600px] pr-4">
              <div className="space-y-6">
                {/* Weather Data */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Cloud className="h-5 w-5 text-primary" />
                      Weather Data
                    </CardTitle>
                    <CardDescription>
                      Current and historical weather conditions
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label htmlFor="rainfall">Rainfall (mm)</Label>
                      <Input
                        id="rainfall"
                        type="number"
                        placeholder="0.0"
                        value={formData.rainfall}
                        onChange={(e) => handleInputChange('rainfall', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="temperature">Temperature (°C)</Label>
                      <Input
                        id="temperature"
                        type="number"
                        placeholder="20.0"
                        value={formData.temperature}
                        onChange={(e) => handleInputChange('temperature', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="humidity">Humidity (%)</Label>
                      <Input
                        id="humidity"
                        type="number"
                        placeholder="60"
                        value={formData.humidity}
                        onChange={(e) => handleInputChange('humidity', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="rainfall3Day">3-Day Rainfall Total (mm)</Label>
                      <Input
                        id="rainfall3Day"
                        type="number"
                        placeholder="0.0"
                        value={formData.rainfall3Day}
                        onChange={(e) => handleInputChange('rainfall3Day', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2 md:col-span-2">
                      <Label htmlFor="rainfall7Day">7-Day Rainfall Total (mm)</Label>
                      <Input
                        id="rainfall7Day"
                        type="number"
                        placeholder="0.0"
                        value={formData.rainfall7Day}
                        onChange={(e) => handleInputChange('rainfall7Day', e.target.value)}
                      />
                    </div>
                  </CardContent>
                </Card>

                {/* Topographical Data */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Mountain className="h-5 w-5 text-primary" />
                      Topographical Data
                    </CardTitle>
                    <CardDescription>
                      Terrain and elevation characteristics
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label htmlFor="slopeAngle">Slope Angle (degrees)</Label>
                      <Input
                        id="slopeAngle"
                        type="number"
                        placeholder="15"
                        value={formData.slopeAngle}
                        onChange={(e) => handleInputChange('slopeAngle', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="aspect">Aspect (degrees)</Label>
                      <Input
                        id="aspect"
                        type="number"
                        placeholder="180"
                        value={formData.aspect}
                        onChange={(e) => handleInputChange('aspect', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2 md:col-span-2">
                      <Label htmlFor="elevation">Elevation (m)</Label>
                      <Input
                        id="elevation"
                        type="number"
                        placeholder="500"
                        value={formData.elevation}
                        onChange={(e) => handleInputChange('elevation', e.target.value)}
                      />
                    </div>
                  </CardContent>
                </Card>

                {/* Soil Properties */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Layers className="h-5 w-5 text-primary" />
                      Soil Properties
                    </CardTitle>
                    <CardDescription>
                      Soil composition and characteristics
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label htmlFor="soilSaturation">Soil Saturation (%)</Label>
                      <Input
                        id="soilSaturation"
                        type="number"
                        placeholder="75"
                        value={formData.soilSaturation}
                        onChange={(e) => handleInputChange('soilSaturation', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="soilPh">Soil pH</Label>
                      <Input
                        id="soilPh"
                        type="number"
                        step="0.1"
                        placeholder="6.5"
                        value={formData.soilPh}
                        onChange={(e) => handleInputChange('soilPh', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="clayContent">Clay Content (%)</Label>
                      <Input
                        id="clayContent"
                        type="number"
                        placeholder="30"
                        value={formData.clayContent}
                        onChange={(e) => handleInputChange('clayContent', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="sandContent">Sand Content (%)</Label>
                      <Input
                        id="sandContent"
                        type="number"
                        placeholder="40"
                        value={formData.sandContent}
                        onChange={(e) => handleInputChange('sandContent', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="siltContent">Silt Content (%)</Label>
                      <Input
                        id="siltContent"
                        type="number"
                        placeholder="30"
                        value={formData.siltContent}
                        onChange={(e) => handleInputChange('siltContent', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="soilErosionRate">Soil Erosion Rate</Label>
                      <Input
                        id="soilErosionRate"
                        type="number"
                        step="0.01"
                        placeholder="0.5"
                        value={formData.soilErosionRate}
                        onChange={(e) => handleInputChange('soilErosionRate', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="poreWaterPressure">Pore Water Pressure (kPa)</Label>
                      <Input
                        id="poreWaterPressure"
                        type="number"
                        placeholder="25"
                        value={formData.poreWaterPressure}
                        onChange={(e) => handleInputChange('poreWaterPressure', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="soilMoistureContent">Soil Moisture Content (%)</Label>
                      <Input
                        id="soilMoistureContent"
                        type="number"
                        placeholder="45"
                        value={formData.soilMoistureContent}
                        onChange={(e) => handleInputChange('soilMoistureContent', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="soilTemperature">Soil Temperature (°C)</Label>
                      <Input
                        id="soilTemperature"
                        type="number"
                        placeholder="15"
                        value={formData.soilTemperature}
                        onChange={(e) => handleInputChange('soilTemperature', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="soilStrain">Soil Strain</Label>
                      <Input
                        id="soilStrain"
                        type="number"
                        step="0.001"
                        placeholder="0.002"
                        value={formData.soilStrain}
                        onChange={(e) => handleInputChange('soilStrain', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="tdrReflectionIndex">TDR Reflection Index</Label>
                      <Input
                        id="tdrReflectionIndex"
                        type="number"
                        step="0.01"
                        placeholder="0.75"
                        value={formData.tdrReflectionIndex}
                        onChange={(e) => handleInputChange('tdrReflectionIndex', e.target.value)}
                      />
                    </div>
                    <div className="space-y-3 md:col-span-2">
                      <Label>Soil Type</Label>
                      <div className="grid grid-cols-2 gap-4">
                        {['Gravel', 'Sand', 'Silt', 'Clay'].map((type) => (
                          <div key={type} className="flex items-center space-x-2">
                            <Checkbox
                              id={`soil-${type.toLowerCase()}`}
                              checked={formData.soilType.includes(type)}
                              onCheckedChange={(checked) => 
                                handleSoilTypeChange(type, checked as boolean)
                              }
                            />
                            <Label htmlFor={`soil-${type.toLowerCase()}`}>{type}</Label>
                          </div>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Environmental Data */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Leaf className="h-5 w-5 text-primary" />
                      Environmental Data
                    </CardTitle>
                    <CardDescription>
                      Vegetation and land use characteristics
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label htmlFor="vegetationCover">Vegetation Cover (%)</Label>
                      <Input
                        id="vegetationCover"
                        type="number"
                        placeholder="60"
                        value={formData.vegetationCover}
                        onChange={(e) => handleInputChange('vegetationCover', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="ndviIndex">NDVI Index</Label>
                      <Input
                        id="ndviIndex"
                        type="number"
                        step="0.01"
                        placeholder="0.65"
                        value={formData.ndviIndex}
                        onChange={(e) => handleInputChange('ndviIndex', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="proximityToWater">Proximity to Water (m)</Label>
                      <Input
                        id="proximityToWater"
                        type="number"
                        placeholder="100"
                        value={formData.proximityToWater}
                        onChange={(e) => handleInputChange('proximityToWater', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="distanceToRoad">Distance to Road (m)</Label>
                      <Input
                        id="distanceToRoad"
                        type="number"
                        placeholder="50"
                        value={formData.distanceToRoad}
                        onChange={(e) => handleInputChange('distanceToRoad', e.target.value)}
                      />
                    </div>
                    <div className="space-y-3 md:col-span-2">
                      <Label>Land Use</Label>
                      <RadioGroup
                        value={formData.landUse}
                        onValueChange={(value) => handleInputChange('landUse', value)}
                      >
                        <div className="flex items-center space-x-2">
                          <RadioGroupItem value="urban" id="land-urban" />
                          <Label htmlFor="land-urban">Urban</Label>
                        </div>
                        <div className="flex items-center space-x-2">
                          <RadioGroupItem value="forest" id="land-forest" />
                          <Label htmlFor="land-forest">Forest</Label>
                        </div>
                        <div className="flex items-center space-x-2">
                          <RadioGroupItem value="agriculture" id="land-agriculture" />
                          <Label htmlFor="land-agriculture">Agriculture</Label>
                        </div>
                      </RadioGroup>
                    </div>
                  </CardContent>
                </Card>

                {/* Geological Data */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Activity className="h-5 w-5 text-primary" />
                      Geological Data
                    </CardTitle>
                    <CardDescription>
                      Seismic activity and geological history
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label htmlFor="earthquakeActivity">Earthquake Activity (magnitude)</Label>
                      <Input
                        id="earthquakeActivity"
                        type="number"
                        step="0.1"
                        placeholder="2.5"
                        value={formData.earthquakeActivity}
                        onChange={(e) => handleInputChange('earthquakeActivity', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="historicalLandslideCount">Historical Landslide Count</Label>
                      <Input
                        id="historicalLandslideCount"
                        type="number"
                        placeholder="3"
                        value={formData.historicalLandslideCount}
                        onChange={(e) => handleInputChange('historicalLandslideCount', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="microseismicActivity">Microseismic Activity (events/day)</Label>
                      <Input
                        id="microseismicActivity"
                        type="number"
                        placeholder="15"
                        value={formData.microseismicActivity}
                        onChange={(e) => handleInputChange('microseismicActivity', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="acousticEmission">Acoustic Emission (dB)</Label>
                      <Input
                        id="acousticEmission"
                        type="number"
                        placeholder="45"
                        value={formData.acousticEmission}
                        onChange={(e) => handleInputChange('acousticEmission', e.target.value)}
                      />
                    </div>
                  </CardContent>
                </Card>
              </div>
            </ScrollArea>
          </div>

          {/* Prediction Panel */}
          <div className="lg:col-span-1">
            <div className="sticky top-4">
              <Card>
                <CardHeader>
                  <CardTitle>Risk Assessment</CardTitle>
                  <CardDescription>
                    Click predict to analyze landslide risk
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Button 
                    onClick={handlePredict} 
                    disabled={isLoading}
                    className="w-full"
                    size="lg"
                  >
                    {isLoading ? 'Analyzing...' : 'Predict Risk'}
                  </Button>

                  {result && (
                    <>
                      <Separator />
                      <div className={`rounded-lg border p-4 ${
                        result.risk === 'high' 
                          ? 'border-destructive bg-destructive/5' 
                          : 'border-success bg-success/5'
                      }`}>
                        <div className="flex items-center gap-3">
                          {result.risk === 'high' ? (
                            <AlertTriangle className="h-6 w-6 text-destructive" />
                          ) : (
                            <CheckCircle className="h-6 w-6 text-success" />
                          )}
                          <div>
                            <h3 className={`font-semibold ${
                              result.risk === 'high' ? 'text-destructive' : 'text-success'
                            }`}>
                              {result.risk === 'high' ? 'High Risk' : 'Low Risk'}
                            </h3>
                            <p className="text-sm text-muted-foreground">
                              {result.message}
                            </p>
                          </div>
                        </div>
                        <div className="mt-3">
                          <div className="flex justify-between text-sm">
                            <span>Confidence</span>
                            <span className="font-medium">{result.confidence}%</span>
                          </div>
                          <div className="mt-1 h-2 w-full rounded-full bg-muted">
                            <div 
                              className={`h-2 rounded-full ${
                                result.risk === 'high' ? 'bg-destructive' : 'bg-success'
                              }`}
                              style={{ width: `${result.confidence}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionForm;