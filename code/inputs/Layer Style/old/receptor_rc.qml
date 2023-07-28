<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis maxScale="0" version="3.16.7-Hannover" minScale="1e+08" styleCategories="AllStyleCategories" hasScaleBasedVisibilityFlag="0">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
  </flags>
  <temporal enabled="0" fetchMode="0" mode="0">
    <fixedRange>
      <start></start>
      <end></end>
    </fixedRange>
  </temporal>
  <customproperties>
    <property key="WMSBackgroundLayer" value="false"/>
    <property key="WMSPublishDataSourceUrl" value="false"/>
    <property key="embeddedWidgets/count" value="0"/>
    <property key="identify/format" value="Value"/>
  </customproperties>
  <pipe>
    <provider>
      <resampling enabled="false" zoomedInResamplingMethod="nearestNeighbour" zoomedOutResamplingMethod="nearestNeighbour" maxOversampling="2"/>
    </provider>
    <rasterrenderer opacity="1" alphaBand="-1" nodataColor="" type="paletted" band="1">
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
        <paletteEntry value="1" label="1 - Low Impact" color="#38a800" alpha="255"/>
        <paletteEntry value="2" label="2 - Low/Med Impact" color="#b0e000" alpha="255"/>
        <paletteEntry value="3" label="3 - Med/High Impact" color="#ffaa00" alpha="255"/>
        <paletteEntry value="4" label="4 - High Impact" color="#ff0000" alpha="255"/>
      </colorPalette>
      <colorramp name="[source]" type="randomcolors"/>
    </rasterrenderer>
    <brightnesscontrast gamma="1" contrast="0" brightness="0"/>
    <huesaturation colorizeBlue="128" colorizeOn="0" saturation="0" grayscaleMode="0" colorizeRed="255" colorizeGreen="128" colorizeStrength="100"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
