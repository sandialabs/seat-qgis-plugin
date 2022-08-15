<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.16.7-Hannover" styleCategories="AllStyleCategories" minScale="1e+08" hasScaleBasedVisibilityFlag="0" maxScale="0">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
  </flags>
  <temporal enabled="0" mode="0" fetchMode="0">
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
      <resampling maxOversampling="2" enabled="false" zoomedOutResamplingMethod="nearestNeighbour" zoomedInResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer opacity="1" band="1" classificationMax="4.2676082" alphaBand="-1" type="singlebandpseudocolor" nodataColor="" classificationMin="-0.4669161">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>MinMax</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader labelPrecision="4" classificationMode="1" colorRampType="INTERPOLATED" minimumValue="-0.4669161" maximumValue="4.2676082" clip="0">
          <colorramp name="[source]" type="gradient">
            <prop v="43,131,186,255" k="color1"/>
            <prop v="215,25,28,255" k="color2"/>
            <prop v="0" k="discrete"/>
            <prop v="gradient" k="rampType"/>
            <prop v="0.25;171,221,164,255:0.5;255,255,191,255:0.75;253,174,97,255" k="stops"/>
          </colorramp>
          <item value="-0.4669161" label="-0.4669" alpha="255" color="#2b83ba"/>
          <item value="0.716714975" label="0.7167" alpha="255" color="#abdda4"/>
          <item value="1.90034605" label="1.9003" alpha="255" color="#ffffbf"/>
          <item value="3.083977125" label="3.0840" alpha="255" color="#fdae61"/>
          <item value="4.2676082" label="4.2676" alpha="255" color="#d7191c"/>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast gamma="1" brightness="0" contrast="0"/>
    <huesaturation colorizeOn="0" colorizeRed="255" colorizeGreen="128" saturation="0" colorizeStrength="100" colorizeBlue="128" grayscaleMode="0"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
