/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.onnx4j.model.graph.node;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.onnx4j.Model;
import org.onnx4j.model.graph.node.attributes.FloatAttribute;
import org.onnx4j.model.graph.node.attributes.FloatsAttribute;
import org.onnx4j.model.graph.node.attributes.IntAttribute;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.model.graph.node.attributes.StringAttribute;
import org.onnx4j.model.graph.node.attributes.StringsAttribute;
import org.onnx4j.model.graph.node.attributes.TensorAttribute;
import org.onnx4j.model.graph.node.attributes.TensorsAttribute;
import org.onnx4j.prototypes.OnnxProto3.AttributeProto;

public class Attributes {

	private Map<String, Attribute<?>> attrs = new HashMap<String, Attribute<?>>();

	public Attributes(Model model, List<AttributeProto> attrProtoList) {
		for (AttributeProto attrProto : attrProtoList) {
			this.addAttr(model, attrProto);
		}
	}

	protected void addAttr(Model model, AttributeProto attrProto) {
		String attrName = attrProto.getName();

		if (attrProto.getType().getNumber() != 0) {
			switch (attrProto.getType().getNumber()) {
			case AttributeProto.AttributeType.INT_VALUE:
				this.attrs.put(attrName, new IntAttribute(attrProto));
				break;
			case AttributeProto.AttributeType.INTS_VALUE:
				this.attrs.put(attrName, new IntsAttribute(attrProto));
				break;
			case AttributeProto.AttributeType.FLOAT_VALUE:
				this.attrs.put(attrName, new FloatAttribute(attrProto));
				break;
			case AttributeProto.AttributeType.FLOATS_VALUE:
				this.attrs.put(attrName, new FloatsAttribute(attrProto));
				break;
			case AttributeProto.AttributeType.STRING_VALUE:
				this.attrs.put(attrName, new StringAttribute(attrProto));
				break;
			case AttributeProto.AttributeType.STRINGS_VALUE:
				this.attrs.put(attrName, new StringsAttribute(attrProto));
				break;
			case AttributeProto.AttributeType.TENSOR_VALUE:
				this.attrs.put(attrName, new TensorAttribute(model, attrProto));
				break;
			case AttributeProto.AttributeType.TENSORS_VALUE:
				this.attrs.put(attrName, new TensorsAttribute(model, attrProto));
				break;
			default:
				throw new UnsupportedOperationException(
						String.format("Unable to handle the attribute \"%s\" as \"%s\" type", attrProto.getName(),
								attrProto.getType().name()));
			}
		} else {
			if (attrProto.hasField(AttributeProto.getDescriptor().findFieldByNumber(AttributeProto.I_FIELD_NUMBER))) {
				this.attrs.put(attrName, new IntAttribute(attrProto));
			} else if (attrProto.getIntsCount() > 0) {
				this.attrs.put(attrName, new IntsAttribute(attrProto));
			} else if (attrProto
					.hasField(AttributeProto.getDescriptor().findFieldByNumber(AttributeProto.F_FIELD_NUMBER))) {
				this.attrs.put(attrName, new FloatAttribute(attrProto));
			} else if (attrProto.getFloatsCount() > 0) {
				this.attrs.put(attrName, new FloatsAttribute(attrProto));
			} else if (attrProto
					.hasField(AttributeProto.getDescriptor().findFieldByNumber(AttributeProto.S_FIELD_NUMBER))) {
				this.attrs.put(attrName, new StringAttribute(attrProto));
			} else if (attrProto.getStringsCount() > 0) {
				this.attrs.put(attrName, new StringsAttribute(attrProto));
			} else if (attrProto.hasT()) {
				this.attrs.put(attrName, new TensorAttribute(model, attrProto));
			} else if (attrProto.getTensorsCount() > 0) {
				this.attrs.put(attrName, new TensorsAttribute(model, attrProto));
			} else {
				// Ignore it
			}
		}
	}

	public Attribute<?> getAttr(String attrName) {
		return this.attrs.get(attrName);
	}

	public <T_ATTR extends Attribute<T_VAL>, T_VAL> T_VAL getAttrValue(String attrName,
			Class<? extends T_ATTR> typeOfAttr, T_VAL defValue) {
		Attribute<?> attr = this.attrs.get(attrName);
		if (attr == null)
			return defValue;

		return typeOfAttr.cast(attr).getValue();
	}

}