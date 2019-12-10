package org.onnx4j.model.graph.node;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.onnx4j.Tensor;
import org.onnx4j.model.graph.node.attributes.FloatAttribute;
import org.onnx4j.model.graph.node.attributes.FloatsAttribute;
import org.onnx4j.model.graph.node.attributes.IntAttribute;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.model.graph.node.attributes.NullAttribute;
import org.onnx4j.model.graph.node.attributes.StringAttribute;
import org.onnx4j.model.graph.node.attributes.StringsAttribute;
import org.onnx4j.model.graph.node.attributes.TensorAttribute;
import org.onnx4j.model.graph.node.attributes.TensorsAttribute;
import org.onnx4j.onnx.prototypes.OnnxProto3.AttributeProto;

public class Attributes implements AutoCloseable {

	private Map<String, Attribute<?>> attrs = new HashMap<String, Attribute<?>>();

	public Attributes(List<AttributeProto> attrProtoList, Tensor.Options tensorOptions) {
		for (AttributeProto attrProto : attrProtoList) {
			this.addAttr(attrProto, tensorOptions);
		}
	}

	protected void addAttr(AttributeProto attrProto, Tensor.Options tensorOptions) {
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
				this.attrs.put(attrName, new TensorAttribute(attrProto, tensorOptions));
				break;
			case AttributeProto.AttributeType.TENSORS_VALUE:
				this.attrs.put(attrName, new TensorsAttribute(attrProto, tensorOptions));
				break;
			default:
				this.attrs.put(attrName, new NullAttribute(attrProto));
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
				this.attrs.put(attrName, new TensorAttribute(attrProto, tensorOptions));
			} else if (attrProto.getTensorsCount() > 0) {
				this.attrs.put(attrName, new TensorsAttribute(attrProto, tensorOptions));
			} else {
				// this.attrs.put(attrName, new Attribute<Object>(null));
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

	@Override
	public void close() throws Exception {
		for (Entry<String, Attribute<?>> entry : attrs.entrySet()) {
			entry.getValue().close();
		}
	}
	

}
