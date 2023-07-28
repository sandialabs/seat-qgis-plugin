<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.16.7-Hannover" minScale="1e+08" styleCategories="AllStyleCategories" maxScale="0" hasScaleBasedVisibilityFlag="0">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
  </flags>
  <temporal fetchMode="0" enabled="0" mode="0">
    <fixedRange>
      <start></start>
      <end></end>
    </fixedRange>
  </temporal>
  <customproperties>
    <property value="false" key="WMSBackgroundLayer"/>
    <property value="false" key="WMSPublishDataSourceUrl"/>
    <property value="0" key="embeddedWidgets/count"/>
    <property value="Value" key="identify/format"/>
  </customproperties>
  <pipe>
    <provider>
      <resampling maxOversampling="2" zoomedOutResamplingMethod="nearestNeighbour" enabled="false" zoomedInResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer opacity="1" band="1" alphaBand="-1" type="paletted" nodataColor="">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <colorPalette>
        <paletteEntry value="50" label="50" color="#79de13" alpha="255"/>
        <paletteEntry value="150" label="150" color="#a86ee1" alpha="255"/>
        <paletteEntry value="300" label="300" color="#4defef" alpha="255"/>
        <paletteEntry value="5000" label="5000" color="#dc0f0f" alpha="255"/>
      </colorPalette>
      <colorramp name="[source]" type="randomcolors"/>
    </rasterrenderer>
    <brightnesscontrast gamma="1" brightness="0" contrast="0"/>
    <huesaturation colorizeRed="255" colorizeOn="0" saturation="0" colorizeGreen="128" colorizeBlue="128" grayscaleMode="0" colorizeStrength="100"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
