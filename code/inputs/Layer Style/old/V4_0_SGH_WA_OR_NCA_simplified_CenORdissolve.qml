<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.16.7-Hannover" minScale="1e+08" hasScaleBasedVisibilityFlag="0" styleCategories="AllStyleCategories" maxScale="0">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
  </flags>
  <temporal mode="0" enabled="0" fetchMode="0">
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
      <resampling zoomedInResamplingMethod="nearestNeighbour" enabled="false" zoomedOutResamplingMethod="nearestNeighbour" maxOversampling="2"/>
    </provider>
    <rasterrenderer type="paletted" opacity="1" alphaBand="-1" nodataColor="" band="1">
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
        <paletteEntry value="1" color="#79de13" label="1 - Mixed" alpha="255"/>
        <paletteEntry value="2" color="#a86ee1" label="2 - Mud" alpha="255"/>
        <paletteEntry value="3" color="#4defef" label="3 - Rock" alpha="255"/>
        <paletteEntry value="4" color="#dc0f0f" label="4 - Sand or Gravel" alpha="255"/>
      </colorPalette>
      <colorramp type="randomcolors" name="[source]"/>
    </rasterrenderer>
    <brightnesscontrast gamma="1" brightness="0" contrast="0"/>
    <huesaturation grayscaleMode="0" colorizeGreen="128" colorizeStrength="100" colorizeBlue="128" colorizeRed="255" saturation="0" colorizeOn="0"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
